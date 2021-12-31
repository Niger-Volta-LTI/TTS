import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from pathlib import Path

# set experiment paths
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(output_path, "/Users/rebeiro/github/tts/open-bible-yo/")

# download the dataset if not downloaded
# if not os.path.exists(dataset_path):
#     from TTS.utils.downloaders import download_vctk
#
#     download_vctk(dataset_path)

# define dataset config
dataset_config = BaseDatasetConfig(name="open-bible-yo", meta_file_train="metadata.tsv", path=dataset_path)

# define audio config
# ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
audio_config = BaseAudioConfig(sample_rate=48000, resample=False, do_trim_silence=True, trim_db=23.0)


def bible_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes Open Bible dataset to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    speaker_name = Path(root_path).stem
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("\t")
            wav_file = os.path.join(root_path + "wavs", cols[0].strip())
            text = cols[1]
            if os.path.isfile(wav_file):
                if len(text) > 0:
                    text = text.strip().lower()
                    items.append([text, wav_file, speaker_name])
                else:
                    print("text len <= 0, empty text?")
            else:
                print("> File %s does not exist!" % (wav_file))
    return items


# define model config
config = GlowTTSConfig(
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    audio=audio_config,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True,
                                               formatter=bible_formatter)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.num_speakers = speaker_manager.num_speakers

# init model
model = GlowTTS(config, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
