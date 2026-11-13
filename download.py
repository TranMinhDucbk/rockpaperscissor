import urllib.request, zipfile

bases = [
    "https://storage.googleapis.com/download.tensorflow.org/data/",
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/",
]

for name in ["rps.zip", "rps-test-set.zip"]:
    for base in bases:
        try:
            print("Downloading", base + name)
            urllib.request.urlretrieve(base + name, name)
            break
        except Exception as e:
            print("  Failed:", e)
    else:
        raise SystemExit("Could not download " + name)
    zipfile.ZipFile(name).extractall(".")
print("Done")