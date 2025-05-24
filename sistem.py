import streamlit as st
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gspread
from gspread.exceptions import SpreadsheetNotFound, GSpreadException
from google.oauth2.service_account import Credentials
from datetime import datetime



st.markdown("""
<style>
    .stExpander {
        font-weight: bold;
    }
    .stExpander:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_assets():
    # Load model with custom handler
    model = joblib.load('lightfm_model.pkl')    
    # Load other data
    data = joblib.load('supporting_data.pkl')
    return model, data

model, data = load_assets()

# Access the features like this:
user_features_test = data['user_features_test']
# item_features = data['item_features']
books_df = data['books_df']
users_df = data['users_df']
user_id_mapping = data['user_id_mapping']
item_id_mapping = data['item_id_mapping']
cold_user_ids = data['cold_user_ids']
test_ratings = data['test_ratings']


# At the beginning of your code (where you define user_features_test)
num_users = len(user_id_mapping)

# App layout
st.title("📖 Book Recommendation System")
st.markdown("Discover your next favorite book!")
st.divider()

num_recommendations = st.slider("Number of recommendations", 5, 20, 10)


# Initialize Google Sheets connection
# Updated Google Sheets connection with proper error handling
def init_gsheets():
    try:
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scope
        )
        client = gspread.authorize(creds)
        return client.open("Book_Recommendation_Feedback").sheet1
    except SpreadsheetNotFound:
        st.error("Google Sheet not found. Check the sheet name exists.")
        return None
    except GSpreadException as e:  # General exception catch
        st.error(f"Google Sheets error: {str(e)}")
        return None

# Feedback saving function
def save_feedback(email, rating, feedback_text):
    try:
        sheet = init_gsheets()
        if not sheet:
            return False
            
        # Append row and verify success
        response = sheet.append_row([
            datetime.now().isoformat(),
            email or "anonymous",
            rating,
            feedback_text,
            "v1.0"
        ], value_input_option="USER_ENTERED")
        
        # Check if updates were successful
        if not response.get('updates', {}).get('updatedRows'):
            raise ValueError("No rows were updated in Google Sheets")
            
        return True
        
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False
    

st.markdown("---")
with st.form("recommendation_feedback"):
    st.subheader("📝 Help us improve our recommendations!")
    
    # Email collection (optional)
    email = st.text_input("Email (optional but very much preferred):")
    
    # Rating scale
    rating = st.radio("How relevant were these recommendations?", 
                     ["🤮 Absolutely horrible", "😞 Poor", "😐 Fair", "😀 Good!", "😍 Breathtakingly excellent!"],
                     horizontal=True)
    
    # Detailed feedback
    feedback_text = st.text_area("What could we improve? (also optional)")
    
    # Form submission
    submitted = st.form_submit_button("Submit Feedback")
    
    if submitted:
        # Map ratings to numerical values
        rating_map = {
            "🤮 Horrible": 1,
            "😞 Poor": 2,
            "😐 Fair": 3,
            "😀 Good": 4,
            "😍 Excellent": 5
        }
        
        if save_feedback(
            email=email if email else "anonymous",
            rating=rating,
            feedback_text=feedback_text
        ):
            st.success("🎉 Thanks for your feedback! We'll use this to improve our recommendations.")
            st.balloons()




def get_star_rating(rating):
    if rating == 5 or rating == 0:
        return 3
    else:
        return round(rating / 2)
    
def display_user_profile(user_id):
    """Display a compact user profile in the sidebar"""
    with st.sidebar:
        # Get user data
        user_data = users_df[users_df['User-ID'] == user_id].iloc[0]
        
        # Profile header with avatar
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.subheader(f"User ID: {user_id}")
        
        # Basic info
        st.caption(f"📍 {user_data.get('Location', 'Unknown location')}")
        
        # Favorite genres (collapsed by default)
        with st.expander(f"Top {len(user_data['fav_genres'][:3])} Genres", expanded=False):
            for genre in user_data['fav_genres']:  # Show top 5
                st.markdown(f"- {genre}")
        
        # Favorite authors (collapsed by default)
        with st.expander(f"Top {len(user_data['fav_authors'][:3])} Authors", expanded=False):
            for author in user_data['fav_authors']:  # Show top 3
                st.markdown(f"- {author}")
        

# Add this near the top of your app, after loading assets but before user selection
st.sidebar.title("User Selection Mode")
# Radio button to choose between modes
selection_mode = st.sidebar.radio(
    "Choose input method:",
    ["Use Cold-Start Sample", "Enter My Own Preferences"],
    index=0
)

if selection_mode == "Use Cold-Start Sample":
    # Existing cold-user selection
    selected_user = st.selectbox(
        "Select a Cold-Start User Sample:",
        cold_user_ids[17:26]
    )
    
    st.sidebar.divider()
    
    # Display the selected user's profile
    display_user_profile(selected_user)
    
else:
    # Create a form for manual preference input
    with st.sidebar.form("user_preferences"):
        st.subheader("Enter Your Preferences")
        
        # Multi-select for genres (using unique genres from books_df)
        all_genres = sorted(data['genre_feature_mapping'].keys())
        all_authors = sorted(data['author_feature_mapping'].keys())


        selected_genres = st.multiselect(
            "Select your favorite genres:",
            options=all_genres,
            default=[]
        )
        
        # Multi-select for authors
        selected_authors = st.multiselect(
            "Select your favorite authors:",
            options=all_authors,
            default=[]
        )
        
        # Submit button
        submitted = st.form_submit_button("Save Preferences")



if st.button("Generate Recommendations", type="primary", use_container_width=True):
    if selection_mode == "Use Cold-Start Sample":
        user_internal_id = user_id_mapping[selected_user]
        current_user_features = user_features_test[user_internal_id]
        # Generate predictions
        scores = model.predict(
            user_ids=user_internal_id,
            item_ids=np.arange(len(item_id_mapping)),
            user_features=user_features_test,
            num_threads=4
        )
    else:
        # Get dataset mappings from loaded data
        user_feature_map = data['user_feature_map']  # Add this to your saved data
        
        user_feature_vec = np.zeros(len(user_feature_map))
        
        # Set features using direct mapping
        for genre in selected_genres:
            feature_name = f"genre_{genre.strip()}"
            if feature_name in user_feature_map:
                user_feature_vec[user_feature_map[feature_name]] = 1.0

        for author in selected_authors:
            feature_name = f"author_{author.strip()}"
            if feature_name in user_feature_map:
                user_feature_vec[user_feature_map[feature_name]] = 1.0


        # Convert to sparse format
        custom_user_features = sp.csr_matrix(user_feature_vec.reshape(1, -1))

        # Generate predictions for all items
        scores = model.predict(
            user_ids=0,  # Use dummy ID since we're providing features directly
            item_ids=np.arange(len(item_id_mapping)),
            user_features=custom_user_features,
            num_threads=4
        )


    # Get all item IDs
    all_item_ids = list(item_id_mapping.keys())
    
    if selection_mode == "Use Cold-Start Sample":
        # Get top N items
        top_items = np.argsort(-scores)[:num_recommendations]
        recommended_isbns = [all_item_ids[item] for item in top_items]
        
        # Get actual interactions from test set
        actual_books = test_ratings[test_ratings['User-ID'] == selected_user]
        actual_isbns = actual_books['ISBN'].tolist()

        # Calculate overlap between recommendations and actual
        overlap = set(recommended_isbns) & set(actual_isbns)
        st.divider()
        
        # Metrics row
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommendation Accuracy", 
                f"{len(overlap)}/{len(actual_isbns)} matches",
                help="Number of recommended books that user actually interacted with in the test set")
        with col2:
            st.metric("Recommendation Count", num_recommendations, help="Top 10 recommendations")
        
        st.divider()
        
        # Main content columns
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("📚 Model Recommendations", divider="rainbow")
            for i, isbn in enumerate(recommended_isbns, 1):
                book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
                match_indicator = " ✅" if isbn in overlap else ""
                
                # Use expander for each book
                with st.expander(f"{i}. {book_info['Book-Title']}{match_indicator}", expanded=(i==1)):
                    # Display book cover if available (you'd need this in your data)
                    if pd.notna(book_info.get('Image-URL-L')):
                        st.image(book_info['Image-URL-L'], width=100)
                    else:
                        st.image("https://placehold.co/150x200?text=No+Cover", width=100)
                    
                    st.markdown(f"""
                        **📝 Author**: {book_info['Book-Author']}  
                        **🏷️ Genres**: {', '.join(book_info['genres']) if isinstance(book_info['genres'], list) else book_info['genres']}  
                        **📅 Year**: {book_info['Year-Of-Publication']}  
                        **🏛️ Publisher**: {book_info['Publisher']}
                    """)
                    
        
        with col2:
            st.subheader("📖 Actual Interactions in Testing Set", divider="rainbow")
            if len(actual_isbns) > 0:
                for i, isbn in enumerate(actual_isbns, 1):
                    book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
                    rating = actual_books[actual_books['ISBN'] == isbn]['Book-Rating'].values[0]
                    if rating == 0:
                        rating = 5
                    
                    # Use a card-like container
                    with st.expander(f"{i}. {book_info['Book-Title']}", expanded=(i==1)):       
                        if pd.notna(book_info.get('Image-URL-L')):
                            st.image(book_info['Image-URL-L'], width=100)
                        else:
                            st.image("https://placehold.co/150x200?text=No+Cover", width=100)
                                 
                        # Visual rating (stars)
                        stars = "⭐" * get_star_rating(rating)
                        st.markdown(f"**Rating**: {stars} ({rating}/10)")
                        
                        st.markdown(f"""
                        **📝 Author**: {book_info['Book-Author']}  
                        **🏷️ Genres**: {', '.join(book_info['genres']) if isinstance(book_info['genres'], list) else book_info['genres']}  
                        **📅 Year**: {book_info['Year-Of-Publication']}  
                        **🏛️ Publisher**: {book_info['Publisher']}
                        """)
            else:
                st.warning("No recorded interactions for this user in test set")


    else:
        top_indices = np.argsort(-scores)[:num_recommendations]
        isbn_list = list(item_id_mapping.keys())
        recommended_isbns = [isbn_list[idx] for idx in top_indices]

        # Display recommendations
        st.subheader("🎯 Personalized Recommendations")
        st.divider()

        for i, isbn in enumerate(recommended_isbns, 1):
            book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
            
            # Create a card-like container
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if pd.notna(book_info.get('Image-URL-L')):
                        st.image(book_info['Image-URL-L'], use_container_width=True)
                    else:
                        st.image("https://placehold.co/150x200?text=No+Cover", use_container_width=True)
                
                with col2:
                    # Header with title and position
                    st.markdown(f"### {i}. {book_info['Book-Title']}")
                    
                    # Main book info
                    st.markdown(f"**📚 Author**: {book_info['Book-Author']}")
                    
                    # Format genres
                    genres = book_info['genres']
                    if isinstance(genres, list):
                        genres = ", ".join(genres)
                    st.markdown(f"**🏷️ Genres**: {genres}")
                    
                    # Series information
                    if book_info['Series'] and book_info['Series'] != "Standalone":
                        st.markdown(f"**📖 Series**: {book_info['Series']}")
                    
                    # Combined year and publisher
                    st.markdown(f"""
                    **📅 Published**: {book_info['Year-Of-Publication']}  
                    **🏛️ Publisher**: {book_info['Publisher']}
                    """)
                    
                    # Matches highlights
                    matches = []
                    if any(genre in selected_genres for genre in book_info['genres']):
                        matches.append("Genre match!")
                    if book_info['Book-Author'] in selected_authors:
                        matches.append("Author match!")
                    
                    if matches:
                        st.success("✨ " + " • ".join(matches))

            st.divider()
