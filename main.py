from fastapi import FastAPI, Request
import pickle
import numpy as np

# загрузите модель из файла выше
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# создаём приложение FastAPI
app = FastAPI(title="uplift")

@app.post("/predict")
async def predict(request: Request):

		# все данные передаются в json
    data = await request.json()

		# признаки лежат в features, в массиве
    # извлекаем и преобразуем признаки
    features = data["features"]
    features = np.array(features).reshape(1, -1)

    # получаем предсказания
    prediction = model.predict(features)[0][0]

    return {"predict": prediction.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)