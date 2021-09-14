import json
import urllib.request
import urllib.error
import os
import tqdm
im_dir = 'data/images'


def download_image(im_url, path):
    # setting filename and image URL
    # Adding information about user agent
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-Agent',
    #                       'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    # urllib.request.install_opener(opener)

    # calling urlretrieve function to get resource
    try:
        urllib.request.urlretrieve(im_url, path)
    except urllib.error.HTTPError as e:
        print('Cant download ', path)


def run():
    dtypes = ['train', 'val', 'test']
    for type in dtypes:
        print('----{}-----'.format(type))
        with open('data/annotations/uitviic_captions_{}2017.json'.format(type), encoding='utf8') as f:
            image_list = json.load(f)['images']
            for im in tqdm.tqdm(image_list):
                im_path = os.path.join(im_dir, type, im['file_name'])
                if os.path.exists(im_path):
                    print("Skipping", im_path)
                else:
                    download_image(im['coco_url'], im_path)

if __name__ == '__main__':
    run()