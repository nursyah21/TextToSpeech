print("load all library...\n")

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
from pydub import AudioSegment
from pydub.playback import play

print("preparing tokenizer and model...\n")

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")


def test_1():
    print("processing speech to text...\n")

    # tokenize
    input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1
    # retrieve logits
    logits = model(input_values).logits
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    # tokenize
    input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    test_audio = ds[0]['audio']['path']
    print(f"playing audio\n")
    song = AudioSegment.from_file(test_audio, format='flac')
    play(song)
    
    print("result:", transcription,"\n")

def test_2():
    print("processing speech to text...\n")

    # using custom audio
    # convert audio to array
    path_audio = './yt2.wav'
    audio = AudioSegment.from_file(path_audio)
    x = torch.FloatTensor(audio.get_array_of_samples())

    # tokenize
    input_values = processor(x, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    
    print(f"playing audio\n")
    song = AudioSegment.from_file(path_audio, format='flac')
    play(song)
    
    print("result:", transcription,"\n")

if __name__ == "__main__":
    test_1()
    test_2()