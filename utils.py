import sys
from os import listdir, path
from os.path import isfile, join

def fetchModel(modelName):
    import re
    modelName = str(modelName).lower()
    regexMatch = re.compile(f'{modelName}.+\.h5')
    modelRootDir = "models/trained/"
    for f in listdir(modelRootDir):
        if isfile(join(modelRootDir, f)) and regexMatch.search(f):
            return join(modelRootDir, f)

    return None

def getImages(directory):
    validExtensions = [".png", ".jpeg", "jpg"]
    directories = list()
    for f in listdir(directory):
        if not isfile(join(directory, f)):
            continue

        extension = path.splitext(f)

        if len(extension) == 2 and str(extension[1]).lower() in validExtensions:
            directories.append(directory+f)

    return directories

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)
