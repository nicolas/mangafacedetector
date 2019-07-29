import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles
from PIL import Image as PILImage
import base64

export_file_url = 'https://drive.google.com/uc?export=download&id=1cURNuwKbLb0kQm1abeJ2r6xeHbvhCo47'
export_file_name = 'export.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


def crop(path, bbox):
    image = plt.imread(path)
    x0 = int(bbox[0])
    y0 = int(bbox[1])
    width = int(bbox[2])
    height = int(bbox[3])
    return image[x0:x0+width, y0:y0+height, :]


class StubbedObjectCategoryList(ObjectCategoryList):
    def analyze_pred(self, pred): return [
        pred.unsqueeze(0), torch.ones(1).long()]


def loss_fn(preds, targs, class_idxs):
    return L1Loss()(preds, targs.squeeze())


class FaceDetector(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.head = create_head(num_features_model(self.cnn) * 2, 4)

    def forward(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return 2 * (x.sigmoid_() - 0.5)


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await(img_data['file'].read())
    tfms = get_transforms(do_flip=False, max_rotate=0, max_zoom=1,
                          max_lighting=0, max_warp=0, p_affine=1, p_lighting=1)
    img = open_image(BytesIO(img_bytes)).apply_tfms(
        tfms[0], size=224, resize_method=ResizeMethod.SQUISH)

    img.save('./temp.jpeg')
    pred = learn.predict(img)[0].data[0][0]
    im = crop('./temp.jpeg', (pred + 1) / 2 * 224)
    resp_bytes = BytesIO()
    pil_image = PILImage.fromarray(im).save(resp_bytes, format='png')

    img_str = base64.b64encode(resp_bytes.getvalue()).decode()
    img_str = "data:image/png;base64," + img_str
    return JSONResponse({'img': img_str})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0',
                    port=5000, log_level="info")
