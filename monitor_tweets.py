# monitor_combined_tweets_recent_ocr_fixed.py

import os
import sys
import random
import time
import traceback # For error details
import io # For handling image bytes
from datetime import datetime, timedelta, timezone # Keep for potential future use, though not needed for recent search

# --- Project Root Setup (Optional: Keep if needed for local runs) ---
# try:
#     PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT); print(f"Added {PROJECT_ROOT} to sys.path")
# except NameError: print("Running in environment without __file__.")

# --- Essential Imports ---
import pandas as pd
import numpy as np
import requests
import tweepy
import torch
import joblib

# --- Google Cloud Vision Import ---
try: from google.cloud import vision
except ModuleNotFoundError: print("ERROR: Missing 'google-cloud-vision'. Install & set up auth."); sys.exit(1)

# --- Hugging Face Imports ---
try: from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError: print("ERROR: Missing 'transformers'. Install it."); sys.exit(1)

# --- Configuration ---
# ==============================================================================
# <<< USER CONFIGURATION >>>
# ==============================================================================
# --- Google Cloud Credentials ---
# Set os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/key.json" BEFORE running
# Ensure Cloud Vision API billing is ENABLED in GCP project.

# --- Model Path ---
SAVED_MODEL_DIR = "models/models/transformer_cyberbullying_model" # Path where fine-tuned model was saved

# --- Twitter API Credentials ---
# For FETCHING Tweets (using v2 Bearer Token - Standard Access OK for Recent Search & User Timeline)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAHC%2B0AEAAAAA17yWETTecIRQWxxVMZIFZzSCzE8%3DCCkjsqSTeiWnx6eflXb0eocL0hKADIYVB8C5tgrEALjhkfcegj"

# For SENDING DMs (using v1.1 OAuth - NEW ACCOUNT 'Kavindi5081')
# !!! WARNING: HIGH RISK OF ACCOUNT SUSPENSION FOR AUTOMATED DMS !!!
DM_CONSUMER_KEY = "HIrErLogB46tdkhzALslBKf9j"
DM_CONSUMER_SECRET = "Sku2k0gRn5GM2aQ2IH29zlP4dzslpboNaxrYVzBNRVOu7Ph6M2"
DM_ACCESS_TOKEN = "1238302849034891265-nEeNe5koRwhGwfBNjRYNBhSWrG7fRl"
DM_ACCESS_TOKEN_SECRET = "c2K9HqEq8pJ6g76dasty8nf2ckGKJfoVzBM6wyX61nrQK"

# --- User IDs & Usernames ---
TARGET_USERNAME_TO_MONITOR = "un6234746"
# User ID to send the alert DM TO (Likely the monitored account itself, ensure this ID is correct)
TARGET_USER_ID_FOR_DM = "1904790453213683712"

# --- Monitoring Configuration ---
CLASSIFICATION_THRESHOLD = 0.7
CYBERBULLYING_CLASS_INDEX = 1
# Fetch limits for EACH source (own tweets AND replies/mentions)
MAX_TWEETS_PER_SOURCE = 20 # Max recent tweets to fetch per category (min 5 for own, min 10 for search)

# ==============================================================================
# <<< END OF USER CONFIGURATION >>>
# ==============================================================================

# --- Guidance Dictionary (Ensure keys match analyze_text output) ---
guidance_dict = {
    "aggressive": ["Aggressive guidance... (CERT info)"],
    "insulting": ["Insulting guidance... (CERT info)"],
    "non-cyberbullying": ["Non-cyberbullying guidance..."],
    "threatening": ["Threatening guidance... (CERT info)"],
    "sexual_harassment": ["Sexual harassment guidance... (CERT info)"]
}

# --- Helper Functions ---
def generate_guidance(classification_label):
    default_guidance = "Remember to prioritize your safety and well-being online. Resources are available if you need support."
    messages = guidance_dict.get(classification_label.lower(), [default_guidance])
    if not messages: messages = [default_guidance]
    if classification_label.lower() not in guidance_dict: print(f"Warning: No specific guidance found for label: '{classification_label}'. Using default.")
    return random.choice(messages)

def generate_twitter_dm(classification_label, probability_score, is_cyberbullying, original_text, included_image=False, author_username=None, is_own_tweet=False):
    guidance = generate_guidance(classification_label)
    if is_own_tweet: dm_message_parts = ["Hey there! Our system analyzed one of your recent tweets."]
    else:
        dm_message_parts = ["Hey there! Our system analyzed a recent reply/mention directed at you"]
        if author_username: dm_message_parts[0] += f" from @{author_username}."
        else: dm_message_parts[0] += "."
    if included_image: dm_message_parts[0] += " (It included an image)."
    if guidance: dm_message_parts.append(guidance)
    if is_cyberbullying: dm_message_parts.append(f"\n(System flagged this as potentially concerning with score {probability_score*100:.0f}%)")
    dm_message_parts.append("\nStay safe and take care. 💙")
    return "\n".join(dm_message_parts)

def send_dm(consumer_key, consumer_secret, access_token, access_token_secret, recipient_user_id, message):
    print(f"--- Attempting DM to {recipient_user_id} using configured credentials ---")
    print("!!! WARNING: Automated DM sending is risky and may violate Twitter policy !!!")
    try:
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        api = tweepy.API(auth); api.send_direct_message(recipient_user_id, text=message)
        print(f"DM sent successfully to user ID {recipient_user_id}!")
        return True
    except tweepy.errors.Forbidden as e: print(f"Error sending DM: Forbidden (403). Check API permissions, DM settings, account suspension, or if receiver blocked sender. Error details: {e}"); return False
    except tweepy.errors.TweepyException as e: print(f"Error sending DM: A Tweepy error occurred: {e}"); return False
    except Exception as e: print(f"Error sending DM: An unexpected error occurred: {e}"); traceback.print_exc(); return False

def extract_text_from_image_url(image_url):
    print(f"  Attempting OCR for image: {image_url}")
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ or not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
            print("  Skipping OCR: GOOGLE_APPLICATION_CREDENTIALS not set or file not found."); return None
        response = requests.get(image_url, stream=True, timeout=15); response.raise_for_status(); image_content = response.content
        client = vision.ImageAnnotatorClient(); image = vision.Image(content=image_content)
        ocr_response = client.text_detection(image=image)
        if ocr_response.error.message: print(f"  Vision API Error: {ocr_response.error.message}"); return None
        if ocr_response.text_annotations: full_text = ocr_response.text_annotations[0].description; print(f"  OCR Extracted Text: '{full_text[:100]}...'"); return full_text.strip()
        else: print("  No text detected by OCR."); return ""
    except requests.exceptions.RequestException as e: print(f"  Error downloading image {image_url}: {e}"); return None
    except NameError: print("  Skipping OCR: google.cloud.vision library not loaded correctly."); return None
    except Exception as e: print(f"  Error during OCR processing for {image_url}: {e}"); traceback.print_exc(); return None

def analyze_text(text: str, model, tokenizer, device, label_encoder=None):
    if not text or not text.strip(): print("  Analysis skipped: Input text is empty."); return None, 0.0
    try:
        model.eval(); inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = model(**inputs); logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prob_cyberbullying = probs.squeeze(0)[CYBERBULLYING_CLASS_INDEX].item() if probs.ndim > 1 else probs[CYBERBULLYING_CLASS_INDEX].item()
        predicted_index = torch.argmax(probs, dim=-1).squeeze().item()
        if predicted_index == 0: predicted_label_str = "non-cyberbullying"
        elif predicted_index == 1: predicted_label_str = "aggressive"
        else: predicted_label_str = f"unknown_class_{predicted_index}"; print(f"Warning: Model predicted unexpected index {predicted_index}")
        return predicted_label_str, prob_cyberbullying
    except Exception as e: print(f"Error during text analysis: {e}"); traceback.print_exc(); return None, 0.0

def get_user_id(username, bearer_token):
    print(f"Fetching user ID for username: @{username}"); user_lookup_url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        response = requests.get(user_lookup_url, headers=headers, timeout=10); response.raise_for_status()
        user_data = response.json(); user_id = user_data.get("data", {}).get("id")
        if not user_id: print(f"Error: Could not find user ID for '@{username}'."); return None
        print(f"Found User ID: {user_id} for @{username}"); return user_id
    except requests.exceptions.RequestException as e: print(f"Error looking up user ID for @{username}: {e}"); return None
    except Exception as e: print(f"Unexpected error during user lookup for @{username}: {e}"); return None

# --- CORRECTED fetch_own_recent_tweets ---
def fetch_own_recent_tweets(user_id, bearer_token, max_results=10):
    """Fetches recent tweets posted BY the specified user ID."""
    if not user_id:
        print("Error: Invalid user_id provided for fetching own tweets.")
        return None, None, None
    print(f"\n--- Fetching own recent tweets for User ID: {user_id} ---")
    tweet_fetch_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    query_params = {
        'max_results': max(5, max_results), # API minimum is 5 if max_results specified
        'tweet.fields': 'id,text,created_at,attachments,author_id',
        'expansions': 'attachments.media_keys,author_id',
        'media.fields': 'url,type',
        'user.fields': 'id,username'
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = None # Initialize response
    try:
        response = requests.get(tweet_fetch_url, headers=headers, params=query_params, timeout=15)
        print(f"Request URL (Own Tweets): {response.url}")
        response.raise_for_status()
        data = response.json(); tweets = data.get("data", [])
        includes = data.get("includes", {}); includes_media = includes.get("media", []); includes_users = includes.get("users", [])
        print(f"Fetched {len(tweets)} own tweets.")
        media_lookup = {media['media_key']: media for media in includes_media}
        user_lookup = {user['id']: user for user in includes_users}
        return tweets, media_lookup, user_lookup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching own tweets: {e}")
        # Safely try to print response details
        try:
            if response is not None:
                 print(f"Response Status Code: {response.status_code}")
                 if hasattr(response, 'text'):
                     print(f"Response Text on Error: {response.text}")
        except Exception as print_exc:
            print(f"Could not print response details: {print_exc}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error fetching own tweets: {e}")
        traceback.print_exc()
        return None, None, None

# --- CORRECTED fetch_recent_replies_mentions ---
def fetch_recent_replies_mentions(target_username, bearer_token, max_results=10):
    """Fetches recent tweets replying to or mentioning the target user."""
    print(f"\n--- Fetching recent replies/mentions for @{target_username} (~last 7 days) ---")
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query = f"to:{target_username} -is:retweet"
    print(f"Using Search Query: {query}")
    query_params = { 'query': query, 'max_results': max(10, max_results), # API minimum is 10
                     'tweet.fields': 'id,text,created_at,attachments,author_id',
                     'expansions': 'attachments.media_keys,author_id',
                     'media.fields': 'url,type', 'user.fields': 'id,username' }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = None # Initialize response
    try:
        response = requests.get(search_url, headers=headers, params=query_params, timeout=15)
        print(f"Request URL (Replies/Mentions): {response.url}")
        response.raise_for_status(); data = response.json(); tweets = data.get("data", [])
        includes = data.get("includes", {}); includes_media = includes.get("media", []); includes_users = includes.get("users", [])
        print(f"Fetched {len(tweets)} recent replies/mentions.")
        media_lookup = {media['media_key']: media for media in includes_media}
        user_lookup = {user['id']: user for user in includes_users}
        return tweets, media_lookup, user_lookup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tweets via recent search: {e}")
        # Safely try to print response details
        try:
            if response is not None:
                 print(f"Response Status Code: {response.status_code}")
                 if hasattr(response, 'text'):
                     print(f"Response Text on Error: {response.text}")
        except Exception as print_exc:
            print(f"Could not print response details: {print_exc}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error fetching recent search: {e}")
        traceback.print_exc()
        return None, None, None

# --- process_tweets Function (Keep as before) ---
def process_tweets(tweets_to_process, media_lookup, user_lookup, monitored_user_id,
                   model, tokenizer, device, label_encoder,
                   dm_consumer_key, dm_consumer_secret, dm_access_token, dm_access_token_secret,
                   recipient_user_id, threshold=0.7):
    if not tweets_to_process: print("No tweets found or provided to process."); return
    if media_lookup is None: media_lookup = {}
    if user_lookup is None: user_lookup = {}
    print("\n--- Processing Combined Tweets ---")
    dm_sent_count = 0; dm_failed_count = 0
    for tweet in tweets_to_process:
        original_text = tweet.get('text', ''); tweet_id = tweet.get('id', 'N/A'); combined_text = original_text
        author_id = tweet.get('author_id'); author_username = user_lookup.get(author_id, {}).get('username', 'Unknown User')
        is_own_tweet = (str(author_id) == str(monitored_user_id)) # Compare as strings for safety

        image_url_processed = None; image_text_extracted = None; has_image = False
        log_prefix = "OWN TWEET" if is_own_tweet else f"Reply/Mention from @{author_username}"
        print(f"\nProcessing Tweet ID: {tweet_id} ({log_prefix})"); print(f"Original Text: {original_text}")
        attachments = tweet.get('attachments')
        if attachments and 'media_keys' in attachments:
            for media_key in attachments['media_keys']:
                media_item = media_lookup.get(media_key)
                if media_item and media_item.get('type') == 'photo':
                    image_url = media_item.get('url')
                    if image_url:
                        print(f"  Found image URL: {image_url}"); has_image = True
                        image_text = extract_text_from_image_url(image_url)
                        if image_text is not None: image_text_extracted = image_text; combined_text += "\n[Image Text]: " + image_text
                        else: print("  OCR failed or skipped, analyzing tweet text only.")
                        image_url_processed = image_url; break
                    else: print("  Found photo media key but no URL.")
        print(f"Text to analyze: '{combined_text[:150]}...'")
        predicted_label, probability = analyze_text(combined_text, model, tokenizer, device, label_encoder)
        if predicted_label is None: print("Skipping tweet due to analysis error."); continue
        is_cyberbullying = probability >= threshold
        print(f"Predicted Label: '{predicted_label}', Cyberbullying Probability: {probability*100:.1f}%")
        dm_author = None if is_own_tweet else author_username
        dm_message = generate_twitter_dm(predicted_label, probability, is_cyberbullying, original_text, included_image=has_image, author_username=dm_author, is_own_tweet=is_own_tweet)
        print(f"Generated DM Message:\n{dm_message}")

        # --- DM SENDING BLOCK (Keep uncommented if desired) ---
        print(f"Attempting to send DM to user ID: {recipient_user_id}")
        dm_success = send_dm(dm_consumer_key, dm_consumer_secret, dm_access_token, dm_access_token_secret, recipient_user_id, dm_message)
        if dm_success: dm_sent_count += 1
        else: dm_failed_count += 1
        print("Waiting 5 seconds before next action...")
        time.sleep(5)
        # --- END DM SENDING BLOCK ---
        print("-" * 20)
    print(f"\n--- DM Sending Summary ---"); print(f"Successfully sent: {dm_sent_count}"); print(f"Failed attempts: {dm_failed_count}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Initializing Combined Tweet Monitoring Script with OCR ---")
    # --- Step 1: Check Google Cloud Authentication ---
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
         print("\nWARNING: GOOGLE_APPLICATION_CREDENTIALS env var not set! OCR will fail.")
         print("Set using: os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/key.json'")
         print("Ensure GCP project has billing enabled for Cloud Vision API.")
    # --- Step 2: Load Model and Tokenizer ---
    print(f"\nLoading model and tokenizer from: {SAVED_MODEL_DIR}")
    if not os.path.exists(SAVED_MODEL_DIR): print(f"ERROR: Model directory not found: {SAVED_MODEL_DIR}"); sys.exit(1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_DIR); model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)
        print("Model and tokenizer loaded successfully.")
    except Exception as e: print(f"Error loading model/tokenizer: {e}"); traceback.print_exc(); sys.exit(1)
    # --- Step 3: Load Label Encoder (Optional Informational Step) ---
    label_encoder = None; le_filepath = os.path.join(SAVED_MODEL_DIR, 'label_encoder.joblib')
    if os.path.exists(le_filepath):
        try: label_encoder = joblib.load(le_filepath); print(f"LabelEncoder loaded from {le_filepath}"); print(f"Original encoded classes found: {label_encoder.classes_}")
        except Exception as e: print(f"Warning: Could not load LabelEncoder: {e}"); label_encoder = None
    else: print("LabelEncoder file not found. Using direct index-to-string mapping in analysis.")
    # --- Step 4: Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); print(f"Using device: {device}")

    # --- Step 5: Get Target User ID ---
    monitored_user_id = get_user_id(TARGET_USERNAME_TO_MONITOR, BEARER_TOKEN)
    if not monitored_user_id: print(f"Could not retrieve User ID for {TARGET_USERNAME_TO_MONITOR}. Exiting."); sys.exit(1)

    # --- Step 6: Fetch Both Own Tweets and Replies/Mentions ---
    own_tweets, own_media, own_users = fetch_own_recent_tweets(user_id=monitored_user_id, bearer_token=BEARER_TOKEN, max_results=MAX_TWEETS_PER_SOURCE)
    reply_tweets, reply_media, reply_users = fetch_recent_replies_mentions(target_username=TARGET_USERNAME_TO_MONITOR, bearer_token=BEARER_TOKEN, max_results=MAX_TWEETS_PER_SOURCE)

    # --- Step 7: Combine and Deduplicate Results ---
    print("\n--- Combining and Deduplicating Fetched Tweets ---")
    combined_tweets = []; combined_media = {}; combined_users = {}
    if own_tweets: combined_tweets.extend(own_tweets); print(f"Added {len(own_tweets)} own tweets.")
    if own_media: combined_media.update(own_media)
    if own_users: combined_users.update(own_users)
    if reply_tweets: combined_tweets.extend(reply_tweets); print(f"Added {len(reply_tweets)} replies/mentions.")
    if reply_media: combined_media.update(reply_media)
    if reply_users: combined_users.update(reply_users)
    unique_tweets = []; seen_tweet_ids = set()
    for tweet in combined_tweets:
        tweet_id = tweet.get('id')
        if tweet_id and tweet_id not in seen_tweet_ids: unique_tweets.append(tweet); seen_tweet_ids.add(tweet_id)
    print(f"Total unique tweets to process: {len(unique_tweets)}")

    # --- Step 8: Process Combined Tweets ---
    process_tweets(
        tweets_to_process=unique_tweets, media_lookup=combined_media, user_lookup=combined_users,
        monitored_user_id=monitored_user_id, model=model, tokenizer=tokenizer, device=device,
        label_encoder=label_encoder, dm_consumer_key=DM_CONSUMER_KEY, dm_consumer_secret=DM_CONSUMER_SECRET,
        dm_access_token=DM_ACCESS_TOKEN, dm_access_token_secret=DM_ACCESS_TOKEN_SECRET,
        recipient_user_id=TARGET_USER_ID_FOR_DM, threshold=CLASSIFICATION_THRESHOLD)

    print("\n--- Monitoring script finished processing. ---")