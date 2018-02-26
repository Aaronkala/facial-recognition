const cv = require('opencv4nodejs');
const path = require('path');
const fs = require('fs');
const utils = require('./utils')

const nameMappings = ['Iida', 'Aaron'];
const basePath = './data/imgs';
const imgsPredictPath = path.resolve(basePath, 'predict');
const imgPredictFiles = fs.readdirSync(imgsPredictPath);

const imagesPredict = imgPredictFiles
  // get absolute file path
  .map(file => path.resolve(imgsPredictPath, file))
  // read image
  .map(filePath => cv.imread(filePath))
  // face recognizer works with gray scale images
  .map(img => img.bgrToGray())
  // detect and extract face
  .map(utils.getFaceImage)
  // face images must be equally sized
  .map(faceImg => {
    if (faceImg) {
      return faceImg.resize(80, 80)
    }
    return false
  });

const testImages = imagesPredict

// const eigen = new cv.EigenFaceRecognizer();
// eigen.load('eigendata');
// console.log('eigen:');
// runPrediction(eigen, testImages, nameMappings);

const fisher = new cv.FisherFaceRecognizer();
fisher.load('fisherdata');
console.log('fisher:');
const results = utils.runPrediction(fisher, testImages, nameMappings);
const finished = (msg) => { if (msg) console.log(msg) }
imgPredictFiles.forEach((file, i) => {
  const source = imgsPredictPath + '/' + file
  const target = `data/imgs/people/${results[i].label}/${file}`
  if (results[i]) {
    utils.copyFile(source, target, finished)
  }
})

//fs.createReadStream('test.log').pipe(fs.createWriteStream('newLog.log'));

// const lbph = new cv.LBPHFaceRecognizer();
// lbph.load('lbphdata');
// console.log('lbph:');
// runPrediction(lbph, testImages, nameMappings);