# uvicorn app:app --reload
import base64
import io
import numpy as np
import cv2
import segmentation
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response, StreamingResponse

app = FastAPI()

@app.post("/images/get_initial_area/")
async def get_initial_area(file: UploadFile):
    # recieving image
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, img_clust = segmentation.clustering(img)
    img_resp = segmentation.draw_segmentation(segmentation.thresh_bin(img_clust), img)

    _, encoded_img = cv2.imencode('.PNG', img_resp)
    #encoded_img = base64.b64encode(encoded_img)
    ans = io.BytesIO(encoded_img.tobytes())
    ans.seek(0)

    #return {
    #    'filename': file.filename,
    #    'area': encoded_img
    #}

    return StreamingResponse(content=ans, media_type="image/png")