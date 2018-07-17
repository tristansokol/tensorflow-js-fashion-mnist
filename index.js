// function loadDoc() {
//     let xhttp = new XMLHttpRequest();
//     xhttp.onreadystatechange = function() {
//       if (this.readyState == 4 && this.status == 200) {
//       console.log(this.responseText);
//       }
//     };
//     xhttp.open('GET', 'digits/training-0.json', true);
//     xhttp.send();
//   }
//   loadDoc();

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
// How many examples the model should "see" before making a parameter update.
const BATCH_SIZE = 64;
// How many batches to train the model for.
const TRAIN_BATCHES = 5;

// Every TEST_ITERATION_FREQUENCY batches, test accuracy over TEST_BATCH_SIZE examples.
// Ideally, we'd compute accuracy over the whole test set, but for performance
// reasons we'll use a subset.
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
/**
 *
 *
 */
async function loadData() {
    /**
     *
     *
     * @return {*}
     */
    function loadDigit(file) {
        return new Promise(function(resolve, reject) {
            let xhttp = new XMLHttpRequest();
            xhttp.onreadystatechange = function() {
                if (this.readyState == 4 && this.status == 200) {
                    resolve(JSON.parse(this.responseText));
                }
            };
            xhttp.open('GET', file, true);
            xhttp.send();
        });
    }
    return new Promise(function(resolve, reject) {
        Promise.all([
            loadDigit('digits/training-0.json'),
            loadDigit('digits/training-1.json'),
            loadDigit('digits/training-2.json'),
            loadDigit('digits/training-3.json'),
            loadDigit('digits/training-4.json'),
            loadDigit('digits/training-5.json'),
            loadDigit('digits/training-6.json'),
            loadDigit('digits/training-7.json'),
            loadDigit('digits/training-8.json'),
            loadDigit('digits/training-9.json'),
            loadDigit('digits/test-0.json'),
            loadDigit('digits/test-1.json'),
            loadDigit('digits/test-2.json'),
            loadDigit('digits/test-3.json'),
            loadDigit('digits/test-4.json'),
            loadDigit('digits/test-5.json'),
            loadDigit('digits/test-6.json'),
            loadDigit('digits/test-7.json'),
            loadDigit('digits/test-8.json'),
            loadDigit('digits/test-9.json'),
        ]).then(function(values) {
            let out = {
                training: {
                    0: values[0],
                    1: values[1],
                    2: values[2],
                    3: values[3],
                    4: values[4],
                    5: values[5],
                    6: values[6],
                    7: values[7],
                    8: values[8],
                    9: values[9],
                },
                test: {
                    0: values[10],
                    1: values[11],
                    2: values[12],
                    3: values[13],
                    4: values[14],
                    5: values[15],
                    6: values[16],
                    7: values[17],
                    8: values[18],
                    9: values[19],
                },
            };

            resolve(out);
        });
    });
}

const model = tf.sequential();

model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling',
}));

model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
}));

model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling',
}));

model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2],
}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax',
}));

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);

model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});


/**
 *
 *
 * @param {*} data
 * @param {*} batchSize
 * @return {Tensor}
 */
function randomBatch(data, batchSize, type = 'training') {
    let labels = [];
    let images = [];
    let categoryArray = [];
    while (labels.length < batchSize) {
        labels.push(Math.floor(Math.random() * 10));

        images.push(data[type][labels[labels.length - 1]][Math.floor(Math.random() * data[type][labels[labels.length - 1]].length)]);
    }
    labels.forEach((element) => {
        let tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        tmp[element] = 1;
        categoryArray.push(tmp);
    });
    return {
        images: tf.tensor2d(images, [batchSize, IMAGE_SIZE]),
        labels: tf.tensor2d(categoryArray, [batchSize, NUM_CLASSES]),
    };
}

/**
 *
 *
 * @param {*} data
 */
async function train(data) {
    const lossValues = [];
    const accuracyValues = [];

    for (let i = 0; i < TRAIN_BATCHES; i++) {
        const [batch, validationData] = tf.tidy(() => {
            const batch = randomBatch(data, BATCH_SIZE);
            // debugger;
            batch.xs = batch.images.reshape([BATCH_SIZE, 28, 28, 1]);

            let validationData;
            // Every few batches test the accuracy of the model.
            if (i % TEST_ITERATION_FREQUENCY === 0) {
                const testBatch = randomBatch(data, TEST_BATCH_SIZE, 'test');
                validationData = [
                    // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
                    // that we can feed it to our convolutional neural net.
                    testBatch.images.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels,
                ];
            }
            return [batch, validationData];
        });
        demo = {
            images: tf.tensor2d(batch.images.dataSync().slice(0, 28 * 28), [28, 28]),
            labels: tf.tensor2d(batch.labels.dataSync().slice(0, 10), [1, 10]),
        };
        draw('input', demo.images.dataSync().slice(0, 28 * 28), 28, i ? 1 : 2);

        // The entire dataset doesn't fit into memory so we call train repeatedly
        // with batches using the fit() method.
        const history = await model.fit(
            batch.xs, batch.labels,
            {batchSize: BATCH_SIZE, validationData, epochs: 1});
        // model.summary();
        // console.log(model);
        conv2dConv2D1Weights = await model.getLayer(name = 'conv2d_Conv2D1').getWeights()[0].data();
        // console.log(conv2dConv2D1Weights);
        for (let j = 0; j < model.getLayer(name = 'conv2d_Conv2D1').filters; j++) {
            draw('conv2d1-kernal-' + j, conv2dConv2D1Weights.slice(j * 25, j * 25 + 25), 5, i ? 1 : 10);
        }

        model.getLayer(name = 'conv2d_Conv2D1');

         debugger;
        const loss = history.history.loss[0];
        const accuracy = history.history.acc[0];
        // console.log('L: ' + loss, 'A: ' + accuracy);
        lossValues.push(history.history.loss[0]);
        accuracyValues.push(accuracy);
        // addData(accChart, accuracyValues.length, accuracy);
        addData(lossChart, lossValues.length, loss, accuracy);
        // debugger;
        // Plot loss / accuracy.
        // lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});
        // ui.plotLosses(lossValues);

        // if (validationData != null) {
        //   accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
        //   ui.plotAccuracies(accuracyValues);
        // }

        // Call dispose on the training/test tensors to free their GPU memory.
        tf.dispose([batch, validationData]);

        // tf.nextFrame() returns a promise that resolves at the next call to
        // requestAnimationFrame(). By awaiting this promise we keep our model
        // training from blocking the main UI thread and freezing the browser.
        await tf.nextFrame();
    }
}

/**
 *
 *
 */
async function main() {
    const data = await loadData();
    console.log('data loaded');
    lossChart = lossChart();
    train(data);
}
main();


/**
 *
 *
 * @param {*} id
 * @param {*} data
 * @param {*} width
 */
function draw(id, data, width, scale = 1) {
    ctx = document.getElementById(id).getContext('2d');
    ctx.scale(scale, scale);

    // let canvasData = ctx.createImageData(document.getElementById(id).width, document.getElementById(id).height);
    for (let i = 0; i < data.length; i++) {
        if (data[i] >= 0) {
            ctx.fillStyle = 'rgba(' + data[i] * 255 + ',' + data[i] * 255 + ',' + data[i] * 255 + ',' + 1 + ')';
        } else {
            ctx.fillStyle = 'rgba(' + data[i] * -255 + ',' + data[i] * 0 + ',' + data[i] * 0 + ',' + 1 + ')';
        }
        ctx.fillRect(i % width, Math.floor(i / width), 1, 1);
    }
}


function addData(chart, label, loss, accuracy) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(loss);
    chart.data.datasets[1].data.push(accuracy);

    // chart.data.datasets.forEach((dataset) => {
    //     dataset.data.push(data);
    // });
    chart.update(0);
}

function lossChart(loss) {
    let ctx = document.getElementById('loss').getContext('2d');
    let myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'loss',
                data: [],
                fill: false,
                pointRadius: 0,
            }, {
                label: 'accuracy',
                data: [],
                fill: false,
                borderColor: 'red',
                pointRadius: 0,
            }],
        },
        // options: {
        //     scales: {
        //         yAxes: [{
        //             ticks: {
        //                 beginAtZero: true,
        //             },
        //         }],
        //     },
        // },
    });
    return myChart;
}


