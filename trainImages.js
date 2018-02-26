const cv = require('opencv4nodejs');
const path = require('path');
const fs = require('fs');
const getFaceImage = require('./utils').getFaceImage

const basePath = './data/imgs';
const imgsTrainPath = path.resolve(basePath, 'train');
const nameMappings = ['Iida', 'Aaron'];

const imgTrainFiles = fs.readdirSync(imgsTrainPath);

const imagesTrain = imgTrainFiles
  // get absolute file path
  .map(file => path.resolve(imgsTrainPath, file))
  // read image
  .map(filePath =>
    cv.imread(filePath)
  )
  // face recognizer works with gray scale images
  .map(img => img.bgrToGray())
  // detect and extract face
  .map(getFaceImage)
  // face images must be equally sized
  .map(faceImg => faceImg.resize(80, 80));

const trainImages = imagesTrain

// make labels
const labels = imgTrainFiles
  .map(file => nameMappings.findIndex(name => file.includes(name)));

const eigen = new cv.EigenFaceRecognizer();
eigen.load('eigendata')
eigen.train(trainImages, labels);
eigen.save('eigendata')

const fisher = new cv.FisherFaceRecognizer();
eigen.load('fisherdata')
fisher.train(trainImages, labels);
fisher.save('fisherdata')

const lbph = new cv.LBPHFaceRecognizer();
eigen.load('lbphdata')
lbph.train(trainImages, labels);
lbph.save('lbphdata')