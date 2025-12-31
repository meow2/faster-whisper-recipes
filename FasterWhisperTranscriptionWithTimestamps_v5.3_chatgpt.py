import os
import warnings
import time
from datetime import datetime
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def transcribe_mp3_files():
    folder_path = os.path.dirname(os.path.abspath(__file__))

    print("🤖 faster-whisperモデルをロード中...")
    model = WhisperModel(
        "medium",
        device="cpu",          # GPUがあるなら "cuda"
        compute_type="float32" # 精度重視
    )
    print("✅ モデルロード完了")

    audio_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac"))
    ]

    if not audio_files:
        print("❌ 音声ファイルが見つかりません。")
        return

    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        base_name = os.path.splitext(audio_file)[0]

        # === 日時付ハンドル ===
        now_str = datetime.now().strftime("%Y%m%d-%H%M")
        output_filename = f"{base_name}_{now_str}.txt"
        output_file = os.path.join(folder_path, output_filename)

        print(f"\n🎵 処理開始: {audio_file}")

        start_time = time.time()

        segments, info = model.transcribe(
            audio_path,
            language="ja",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=False,
            no_speech_threshold=0.7,
            chunk_length=30,
        )

        results = []
        full_text = ""

        total_duration = info.duration
        total_chunks = int(total_duration // 30) + 1

        for segment in segments:
            text = segment.text.strip()
            if text:
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                results.append(f"[{start} - {end}] {text}")
                full_text += text + " "

            # ===== 進捗計算 =====
            processed_sec = segment.end
            progress_ratio = processed_sec / total_duration if total_duration > 0 else 0
            progress_pct = progress_ratio * 100

            elapsed = time.time() - start_time
            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed
            else:
                remaining = 0

            current_chunk = int(processed_sec // 30) + 1

            print(
                f"⏳ {progress_pct:6.2f}% | "
                f"チャンク {current_chunk}/{total_chunks} | "
                f"{format_timestamp(processed_sec)} / {format_timestamp(total_duration)} | "
                f"残り約 {format_timestamp(remaining)}",
                end="\r",
                flush=True
            )

        print()  # 改行（進捗行の後）

        with open(output_file, "w", encoding="utf-8-sig") as f:
            f.write("=== 文字起こし結果 ===\n")
            f.write(f"ファイル名: {audio_file}\n")
            f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"検出言語: {info.language}\n")
            f.write(f"音声長: {format_timestamp(total_duration)}\n\n")

            f.write("--- 全文 ---\n")
            f.write(full_text.strip() + "\n\n")

            f.write("--- タイムスタンプ付き ---\n")
            f.write("\n".join(results))

        print(f"✅ 完了: {output_file}")

    print("\n📄 全処理完了")


if __name__ == "__main__":
    transcribe_mp3_files()
