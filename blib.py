import cv2
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # colocando o modelo para trabalhar na brutalidade

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Abre o vídeo
cap = cv2.VideoCapture("/content/istockphoto-1286553323-640_adpp_is.mp4")

# Obtém a taxa de quadros por segundo (FPS) do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)

# Pega o primeiro frame
_, prev_frame = cap.read()

# altura e largura do video original
h, w, _ = prev_frame.shape

# limiar para avaliara a similaridade de um frame a outro
threshold = 0.036

# label previsão do modelo
out_ = "n/a"

text = "planting of"

output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

while cap.isOpened():
  #Lê o próximo frame
  ret, next_frame = cap.read()

  if not ret:
    break

  # Calcula a diferença entre os frames
  difference = cv2.absdiff(prev_frame, next_frame)
  percentage = (difference.sum() / (prev_frame.shape[0] * prev_frame.shape[1] * 3)) / 255

  # Se a diferença for maior que o limite definido
  if percentage > threshold:
    #print(percentage)
    #print(next_frame.shape)
    inputs = processor(next_frame, text, return_tensors="pt").to(device)

    # unconditional image captioning
    #inputs = processor(next_frame, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    out_ = processor.decode(out[0], skip_special_tokens=True)
    #print(out_)

    # Atualiza o frame anterior
    prev_frame = next_frame

  #print(":::::::::::::::",out_)
    cv2.putText(next_frame, out_, (10, next_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  output_video.write(next_frame)

output_video.release()
cap.release()
