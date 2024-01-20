import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

def set_thresholds(df, thresh, stat):
    df[f'{stat}_over_under'] = (df[stat] > thresh).astype(int)
    return df

def calculate_win_pct(record):
    wins, losses = map(int, record.split('-'))
    return wins / (wins + losses) if (wins + losses) > 0 else 0

def train_evaluate_model(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    acc = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    print("Cross-Validation Scores:", scores)
    print("Average Cross-Validation Score:", scores.mean())
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    return model, acc

def calculate_player_stats(player_id, season):
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
    df.sort_values(by='GAME_DATE', inplace=True)
    for stat in ['PTS', 'REB', 'AST']:
        df[f'{stat}_5game_avg'] = df[stat].rolling(window=5).mean()
        df[f'{stat}_std_dev'] = df[stat].rolling(window=5).std()
    df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    df['PRA_5game_avg'] = df['PRA'].rolling(window=5).mean()
    df['PRA_std_dev'] = df['PRA'].rolling(window=5).std()
    df['pts_reb_interaction'] = df['PTS_5game_avg'] * df['REB_5game_avg']
    df['pts_ast_interaction'] = df['PTS_5game_avg'] * df['AST_5game_avg']
    return df

def get_next_game_features(df, model_name):
    latest_stats = df.iloc[-1]
    features = {
        'PTS_5game_avg': latest_stats['PTS_5game_avg'],
        'REB_5game_avg': latest_stats['REB_5game_avg'],
        'AST_5game_avg': latest_stats['AST_5game_avg'],
        'opponent_win_pct': latest_stats['opponent_win_pct'], 
        'player_team_win_pct': latest_stats['player_team_win_pct'],
        f'{model_name}_std_dev': latest_stats[f'{model_name}_std_dev'],
        'pts_reb_interaction': latest_stats['pts_reb_interaction'],
        'pts_ast_interaction': latest_stats['pts_ast_interaction']
    }
    columns = list(features.keys())
    return pd.DataFrame([features], columns=columns)

if __name__ == "__main__":
    player_dict = players.get_players()
    player_name = input("Enter the player's name: ")
    selected_player = [player for player in player_dict if player['full_name'].lower() == player_name.lower()]
    if not selected_player:
        print(f"Player {player_name} not found.")
        exit()
    player_id = selected_player[0]['id']
    player_df = calculate_player_stats(player_id, '2022-23')

    points_thresh = float(input("Enter the threshold for points: "))
    rebounds_thresh = float(input("Enter the threshold for rebounds: "))
    assists_thresh = float(input("Enter the threshold for assists: "))
    pra_thresh = float(input("Enter the threshold for PRA (Points+Rebounds+Assists): "))

    opponent_record = input("Enter the opponent's record (wins-losses): ")
    player_team_record = input("Enter the player's team record (wins-losses): ")

    opponent_win_pct = calculate_win_pct(opponent_record)
    player_team_win_pct = calculate_win_pct(player_team_record)

    player_df['opponent_win_pct'] = opponent_win_pct
    player_df['player_team_win_pct'] = player_team_win_pct

    player_df.dropna(inplace=True)

    stats = ['PTS', 'REB', 'AST', 'PRA']
    thresholds = [points_thresh, rebounds_thresh, assists_thresh, pra_thresh]

    models = {}
    accuracies = {}
    stat_probabilities = {}
    for stat, thresh in zip(stats, thresholds):
        player_df = set_thresholds(player_df, thresh, stat)
        print(f"Model for {stat}:")
        X = player_df[['PTS_5game_avg', 'REB_5game_avg', 'AST_5game_avg', 'opponent_win_pct', 'player_team_win_pct', f'{stat}_std_dev', 'pts_reb_interaction', 'pts_ast_interaction']]
        y = player_df[f'{stat}_over_under']
        model, acc = train_evaluate_model(X, y)
        models[stat] = model
        accuracies[stat] = acc
        print("\n")

    print("Probabilities of Scoring Over for the Next Game:")
    for stat in stats:
        model = models[stat]
        next_game_features = get_next_game_features(player_df, stat)
        prob_over = model.predict_proba(next_game_features)[0, 1]
        stat_probabilities[stat] = prob_over
        print(f"{stat}: {prob_over:.2f}, Accuracy: {accuracies[stat]:.2f}")

    # Identifying the most and least probable stats with high accuracy
    high_accuracy_stats = {stat: prob for stat, prob in stat_probabilities.items() if accuracies[stat] > 0.7}
    if high_accuracy_stats:
        most_probable_stat = max(high_accuracy_stats, key=high_accuracy_stats.get)
        least_probable_stat = min(high_accuracy_stats, key=high_accuracy_stats.get)
        print(f"\nMost probable stat with high accuracy: {most_probable_stat} - Probability: {high_accuracy_stats[most_probable_stat]:.2f}")
        print(f"Least probable stat with high accuracy: {least_probable_stat} - Probability: {high_accuracy_stats[least_probable_stat]:.2f}")
    else:
        print("No stats with high accuracy to report.")
