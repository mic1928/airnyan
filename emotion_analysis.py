import librosa
from openai import OpenAI
import numpy as np

client = OpenAI()


def analyze_emotion(audio_file_path):
    """
    音声ファイルから感情状態を分析する関数

    Args:
        audio_file_path (str): 音声ファイル（mp3）のパス

    Returns:
        tuple: 感情の詳細 (dict), リラックス度 (int)
    """

    # 音声認識 (Whisper APIを使用)
    text = transcribe_audio(audio_file_path)

    # テキスト感情分析 (より詳細な分析)
    text_sentiment = analyze_text_sentiment(text)

    # 音声特徴量分析 (普段の声の高さと比較)
    relaxation = analyze_audio_features(audio_file_path)

    # 結果を辞書にまとめる
    emotion_details = {
        "text_sentiment": text_sentiment,
        "relaxation": relaxation,
    }
    return emotion_details


def transcribe_audio(audio_file_path):
    """
    音声ファイルをテキストに変換する関数 (OpenAI Whisper APIを使用)

    Args:
        audio_file_path (str): 音声ファイル（mp3）のパス

    Returns:
        str: 変換されたテキスト
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        return f"音声認識エラー: {e}"


def analyze_text_sentiment(text):
    """
    テキストから感情を分析する関数 (OpenAI Chat APIを使用)

    Args:
        text (str): 分析対象のテキスト

    Returns:
        dict: 感情分析結果（）
    """
    print(f"テキスト: {text}")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "以下の日本語のテキストの感情の正負を分析してください。0（最も感情が負）から100（最も感情が正）の値を返してください。"},
            {"role": "user", "content": text}
        ]
    )
    sentiment_analysis = response.choices[0].message.content
    return int(sentiment_analysis)


import librosa
import librosa.display

def analyze_audio_features(audio_file_path):
    """
    音声ファイルの声の高さを分析し、リラックス度を推定します。

    Args:
        audio_file_path (str): 音声ファイルのパス

    Returns:
        tuple: 平均ピッチ（Hz）、リラックス度（0〜100）
    """

    y, sr = librosa.load(path=audio_file_path, sr=None)

    # 音量正規化
    y = librosa.util.normalize(y)

    # ピッチを Hz 単位で取得
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # 0 Hz の周波数を除外
    non_zero_indices = np.where(pitches != 0)
    pitches = pitches[non_zero_indices]
    magnitudes = magnitudes[non_zero_indices]

    # 周波数ビンを MIDI 音階に変換 (0 Hz が除外されているので安全)
    pitches = librosa.core.hz_to_midi(pitches) 

    # 音声データの長さ確認
    # print(f"Audio length: {y.size} samples")

    # 大きさが閾値以上のピッチのみ抽出
    valid_pitches = [pitches[i] for i in range(pitches.size) if magnitudes[i] > 0.3]

    # 有効なピッチが存在する場合、平均値と標準偏差を計算
    if valid_pitches:
        average_pitch = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)
    else:
        average_pitch = 0
        pitch_std = 0

    # print(f"平均ピッチ: {average_pitch}")
    # print(f"ピッチの標準偏差: {pitch_std}")

    # リラックス度の目安となる定数（要調整）
    # MAX_STD = 20  # ピッチの標準偏差の最大値（リラックスしていない状態）

    # リラックス度を計算
    # 平均ピッチが60〜70の範囲内にある場合にリラックス度を高くする
    if 60 <= average_pitch <= 70:
        relaxation_base = 100
    else:
        relaxation_base = max(0, 100 - abs(average_pitch - 65))

    # ピッチの標準偏差が小さいほどリラックス度を高くする
    relaxation = relaxation_base - pitch_std
    relaxation = (relaxation - 65) * 10

    # リラックス度を0〜100の範囲に制限
    relaxation = max(0, min(100, relaxation))
    print(f"リラックス度: {relaxation}")

    return relaxation


if __name__ == "__main__":
    # audio_file = "/Users/tomo.f/Desktop/複合現実感システム/airnyan/001-sibutomo.mp3"  # 音声ファイルのパスに置き換える
    # audio_file = "エアにゃん負興奮.m4a" # 78.27, 14.88, 68.44
    # audio_file = "エアにゃん負無興奮.m4a" # 80.66, 15.73, 36.02
    # audio_file = "エアにゃん正興奮.m4a" # 82.30, 15.51, 21.80
    audio_file = "エアにゃん正無興奮.m4a" # 81.00, 15.67, 33.21
    # audio_file = "エアにゃんテスト.m4a" # 83.36, 13.56, 68.07
    # audio_file = "002_AI声優-朱花（のーまるv2）_中国の歌番組で、新....wav"
    
    state= analyze_emotion(audio_file)
    print(f"感情の正負: {state['text_sentiment']}, リラックス度: {state['relaxation']}")


