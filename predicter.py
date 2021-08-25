from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from joblib import load

model = load("model.h5")
categories = ['cats', 'dogs', 'panda']

while True:
    url=input('Enter URL of Image :')

    img=imread(url)
    plt.tick_params()
    plt.title("Image Given")
    plt.imshow(img)
    plt.show()

    img_resize=resize(img,(48, 48, 3))
    l=[img_resize.flatten()]

    print("")

    probability=model.predict_proba(l)
    for ind,val in enumerate(categories):
        print(f'{val} = {probability[0][ind]*100}%')

    print("")

    print("The predicted image is : "+categories[model.predict(l)[0]])

    print("")

    confirmation = str(input("New Image[Y/N]: ")).lower()
    if confirmation == "y":
        continue
    else:
        break