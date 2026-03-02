import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(train_path, test_path, prior_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    prior_df = pd.read_csv(prior_path)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    prior_df.columns = prior_df.columns.str.strip()

    print("Extracting Pure Multi-Horizon Truths...")
    farmer_prior = prior_df.groupby('farmer_name').agg(
        farmer_prior_count=('ID', 'count'),
        farmer_07_rate=('adopted_within_07_days', 'mean'),
        farmer_90_rate=('adopted_within_90_days', 'mean'),
        farmer_120_rate=('adopted_within_120_days', 'mean')
    ).reset_index()

    trainer_prior = prior_df.groupby('trainer').agg(
        trainer_07_rate=('adopted_within_07_days', 'mean'),
        trainer_90_rate=('adopted_within_90_days', 'mean'),
        trainer_120_rate=('adopted_within_120_days', 'mean')
    ).reset_index()

    group_prior = prior_df.groupby('group_name').agg(
        group_07_rate=('adopted_within_07_days', 'mean'),
        group_90_rate=('adopted_within_90_days', 'mean'),
        group_120_rate=('adopted_within_120_days', 'mean')
    ).reset_index()

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    df = df.merge(farmer_prior, on='farmer_name', how='left')
    df = df.merge(trainer_prior, on='trainer', how='left')
    df = df.merge(group_prior, on='group_name', how='left')

    G_07 = prior_df['adopted_within_07_days'].mean()
    G_90 = prior_df['adopted_within_90_days'].mean()
    G_120 = prior_df['adopted_within_120_days'].mean()

    df['farmer_prior_count'] = df['farmer_prior_count'].fillna(0)
    for horizon, G_mean in zip(['07', '90', '120'], [G_07, G_90, G_120]):
        df[f'farmer_{horizon}_rate'] = df[f'farmer_{horizon}_rate'].fillna(G_mean)
        df[f'trainer_{horizon}_rate'] = df[f'trainer_{horizon}_rate'].fillna(G_mean)
        df[f'group_{horizon}_rate'] = df[f'group_{horizon}_rate'].fillna(G_mean)

    print("Processing High-Res NLP & Rescued Behavioral Data...")
    if 'training_day' in df.columns:
        df['training_day_temp'] = pd.to_datetime(df['training_day'], errors='coerce')
        df['training_day_of_week'] = df['training_day_temp'].dt.dayofweek.fillna(-1)
        df.drop(columns=['training_day_temp'], inplace=True)

    topic_col = 'topics_list' if 'topics_list' in df.columns else 'topics'
    if topic_col in df.columns:
        df[topic_col] = df[topic_col].fillna('unknown').astype(str).str.lower()
        df['topic_count'] = df[topic_col].apply(lambda x: len(x.split(',')))
        tfidf = TfidfVectorizer(max_features=50, analyzer='word', token_pattern=r'\w+')
        tfidf_mat = tfidf.fit_transform(df[topic_col])
        tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=[f'tfidf_{i}' for i in range(50)])
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in ['ID', 'farmer_name', 'topics', 'topics_list']:
        if col in cat_cols: cat_cols.remove(col)

    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing')
        df[col] = LabelEncoder().fit_transform(df[col])

    train_clean = df[df['is_train'] == 1].copy()
    test_clean = df[df['is_train'] == 0].copy()

    train_groups = train_clean['farmer_name'].values 
    date_cols = [c for c in train_clean.columns if ('day' in c.lower() or 'date' in c.lower()) and c not in ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days', 'training_day_of_week']]
    targets = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
    leakage_cols = ['ID', 'farmer_name', 'is_train', 'topics', 'topics_list', 'has_second_training'] + targets + date_cols

    X = train_clean.drop(columns=[c for c in leakage_cols if c in train_clean.columns]).fillna(-999)
    X_test = test_clean.drop(columns=[c for c in leakage_cols if c in test_clean.columns]).fillna(-999)
    
    # Store IDs and targets to pass to the model
    test_ids = test_clean[['ID']].copy()
    y_train = train_clean[targets]

    print(f"Features ready! Final X shape: {X.shape}")
    return X, X_test, y_train, train_groups, test_ids