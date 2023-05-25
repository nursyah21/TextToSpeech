# import library

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import speech_recognition as sr
import io
from pydub import AudioSegment

# load model and tokenizer
tokenizer = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def main():

    r = sr.Recognizer()
    print(sr.Microphone.list_microphone_names())
    pass
    with sr.Microphone() as source:
        audio = r.listen(source)  # pyaudio object
        r.recognize_google(audio)
        # while True:
        #     audio = r.listen(source)  # pyaudio object
        #     r.recognize_google(audio)
            # data = io.BytesIO(audio.get_wav_data())  # list of bytes
            # clip = AudioSegment.from_file(data)  # numpy array
            # x = torch.FloatTensor(clip.get_array_of_samples())  # tensor
            #
            # inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
            # logits = model(inputs, axis=-1).logits
            # tokens = torch.argmax(logits)
            # text = tokenizer.batch_decode(tokens)
            #
            # print("You said: ", str(text).lower())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#%%
