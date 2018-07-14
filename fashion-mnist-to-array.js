const fs = require('fs');

/**
 * Adapted from https://github.com/ApelSYN/mnist_dl/blob/master/lib/digitsLoader.js
 * Reads a file and resolves an array of data
 * @param {string} labelFileName
 * @return {promise}
 */
function digitsLoader(labelFileName) {
    return new Promise(function(resolve, reject) {
        digits = [];

        const stream = new fs.ReadStream(labelFileName);
        let ver = 0;
        let start = 0;

        stream.on('readable', function() {
            let buf = stream.read();
            if (buf) {
                if (ver != 2051) {
                    ver = buf.readInt32BE(0);
                    digitCount = buf.readInt32BE(4);
                    start = 16;
                }
                for (let i = start; i < buf.length; i++) {
                    digits.push(buf.readUInt8(i));
                }
                start = 0;
            }
        });

        stream.on('end', function() {
            resolve(digits);
        });
    });
}


/**
 * Adapted from https://github.com/ApelSYN/mnist_dl/blob/master/lib/labelsLoader.js
 * Reads a file and resolves an array of data
 * @param {*} labelFileName
 * @return {promise}
 */
function labelsLoader(labelFileName) {
    return new Promise(function(resolve, reject) {
        labels = [];

        const stream = new fs.ReadStream(labelFileName);
        let ver = 0;
        let start = 0;

        stream.on('readable', function() {
            let buf = stream.read();
            if (buf) {
                if (ver != 2049) {
                    ver = buf.readInt32BE(0);
                    labelCount = buf.readInt32BE(4);
                    start = 8;
                }
                for (let i = start; i < buf.length; i++) {
                    labels.push(buf.readUInt8(i));
                }
                start = 0;
            }
        });

        stream.on('end', function() {
            resolve(labels);
        });
    });
}


/**
 * Adapted from https://github.com/ApelSYN/mnist_dl/blob/master/lib/rawMaker.js
 * Takes arrays of image data and labels and then creates arrays of image data
 * grouped by label.
 * @param {*} labels
 * @param {*} digits
 * @return {array} array of image data indexed by labels
 */
function rawMaker(labels, digits) {
    console.log('Creating ' + labels.length + ' labeled images.');
    let raw = [];
    const imageSize = 28 * 28;
    normalize = function(num) {
        if (num != 0) {
            return Math.round(1000 / (255 / num)) / 1000;
        } else {
            return 0;
        }
    };
    console.log('making labels');
    for (let i in labels) {
        let start = i * imageSize;
        if (!Array.isArray(raw[labels[i]])) {
            raw[labels[i]] = [];
        }
        let range = digits.slice(start, start + imageSize).map(normalize);

        raw[labels[i]].push(...range);
    }
    return raw;
}


/**
 *
 *
 * @param {*} raw
 * @param {*} prefix
 * @param {string} [digitsDir='./digits']
 */
function rawWriter(raw, prefix, digitsDir = './digits') {
    /**
     * adapted from https://blog.raananweber.com/2015/12/15/check-if-a-directory-exists-in-node-js/
     *
     * @param {*} directory
     * @param {*} callback
     */
    function checkDirectory(directory, callback) {
        fs.stat(directory, function(err, stats) {
            // Check if error defined and the error code is "not exists"
            if (err && err.errno === -2) {
                // Create the directory, call the callback.
                fs.mkdir(directory, callback);
            } else {
                // just in case there was a different error:
                callback(err);
            }
        });
    }

    checkDirectory(digitsDir, function(error) {
        console.log(error);
    });
    console.log(raw.length);
    console.log('h');
    for (let i in raw) {
        let out = [];
        for (let j = 0; j < raw[i].length; j = j + 28 * 28) {
            out.push(raw[i].slice(j, j + 28 * 28));
        }
        console.log(out.length);
        let wstream = fs.createWriteStream(`${digitsDir}/${prefix}-${i}.json`);
        // wstream.write('{ "data": [' + raw[i].join(',') + ']}');
        wstream.write(JSON.stringify(out));
        wstream.end();
    }
};


Promise.all([digitsLoader('./data/train-images-idx3-ubyte'),
labelsLoader('./data/train-labels-idx1-ubyte')]).then(function(values) {
    rawWriter(rawMaker(values[1], values[0]), 'training');
});
Promise.all([digitsLoader('./data/t10k-images-idx3-ubyte'),
labelsLoader('./data/t10k-labels-idx1-ubyte')]).then(function(values) {
    rawWriter(rawMaker(values[1], values[0]), 'test');
});
