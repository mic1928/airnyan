from emotion_analysis import analyze_emotion
from cat_reaction import cat_reaction

def simulate_cat_behavior(audio_file_path):
    """
    音声ファイルと飼い主の感情と興奮度に基づいて、猫の行動をシミュレートする関数

    Args:
        audio_file_path (str): 音声ファイル（mp3）のパス
        owner_emotion (int): 飼い主の感情の度合い (0: 負、100: 正)
        owner_excitement (int): 飼い主の興奮度 (0: 低い、100: 高い)

    Returns:
        string: 猫の行動を表す文字列
    """

    # 音声ファイルから感情状態を分析
    emotion_details = analyze_emotion(audio_file_path)
    print(emotion_details)

    # 飼い主の感情と興奮度を考慮して猫の行動を決定
    cat_behavior = cat_reaction(emotion_details["text_sentiment"], emotion_details["relaxation"])

    return cat_behavior

if __name__ == "__main__":
    # テスト用の音声ファイルと飼い主の感情・興奮度
    audio_file_path = "エアにゃん負興奮.m4a"

    # 猫の行動をシミュレート
    cat_behavior = simulate_cat_behavior(audio_file_path)
    print(cat_behavior)