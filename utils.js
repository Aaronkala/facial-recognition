const fs = require('fs')
const cv = require('opencv4nodejs');
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
module.exports = {
  getFaceImage(grayImg) {
    const faceRects = classifier.detectMultiScale(grayImg).objects;
    if (!faceRects.length) {
      console.log('no face detected')
      return false
    }
    return grayImg.getRegion(faceRects[0]);
  },
  runPrediction(recognizer, images, mappings) {
    return images.map((img, i) => {
      if (img) {
        const result = recognizer.predict(img);
        console.log('predicted: %s, confidence: %s', mappings[result.label], result.confidence);
        console.log('result', result)
        return { label: mappings[result.label], confidence: result.confidence }
        //img.pipe(fs.createWriteStream(`data/imgs/people/${result.label}/${i}`))
        //cv.imshowWait('face', img);
        //cv.destroyAllWindows();
      } else {
        console.log('no face')
        return false
      }
    });
  },
  copyFile(source, target, cb) {
    var cbCalled = false;

    var rd = fs.createReadStream(source);
    rd.on("error", function (err) {
      done(err);
    });
    var wr = fs.createWriteStream(target);
    wr.on("error", function (err) {
      done(err);
    });
    wr.on("close", function (ex) {
      done();
    });
    rd.pipe(wr);

    function done(err) {
      if (!cbCalled) {
        cb(err);
        cbCalled = true;
      }
    }
  }
}
