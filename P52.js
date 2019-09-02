const xs = []
const ys = []
let w, b //w는 가중치(기울기),b는 상수(y절편)
const lr = 0.1;//학습정도
const op = tf.train.sgd(lr)

w = tf.variable(tf.scalar(1.0))
b = tf.variable(tf.scalar(0.0))

function loss(predY, realY) {//제곱 후 평균 구함 (경사하강법,mean square 알고리즘)
    return tf.sub(predY, realY).square().mean()//빼기.제곱.평균
}

function predict(x) {
    const tx = tf.tensor1d(x)
    return tf.mul(w, tx).add(b)
}

function mousePressed() {
    let tx = map(mouseX, 0, width, 0, 1)
    let ty = map(mouseY, height, 0, 0, 1)
    // console.log(mouseX, tx, mouseY, ty)
    xs.push(tx)
    ys.push(ty)
}
function setup() {
    createCanvas(windowWidth-100, windowHeight-150)
}
function draw() {
    background(0)
    for (let i = 0; i < xs.length; i++) {
        stroke(255)
        strokeWeight(10)
        let tx = map(xs[i], 0, 1, 0, width)
        let ty = map(ys[i], 0, 1, height, 0)
        point(tx, ty)
    }
    tf.tidy(() => {
        if (xs.length > 0) {
            const realY = tf.tensor1d(ys)
            op.minimize(() => loss(predict(xs), realY))
            let l=loss(predict(xs),realY).dataSync()
            text(l[0],10,10)
        }
        //그림그리기
        const predY = predict([0,1])
        let x1 = 0
        let x2 = width
        // let x1 = map(0, 0, 1, 0, width)
        // let x2 = map(1, 0, 1, 0, width)

        const yy = predY.dataSync()

        let y1 = map(yy[0], 0, 1, height, 0)
        let y2 = map(yy[1], 0, 1, height, 0)
        line(x1, y1, x2, y2)
        // print(x1, y1, x2, y2)
        // console.log('test')
    })
}