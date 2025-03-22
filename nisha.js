// Load the pre-trained TensorFlow.js model
let model;
(async function () {
  model = await tf.loadLayersModel('model/model.json');
  console.log('Model loaded successfully');
})();

// Canvas setup
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Event listeners for drawing
canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
  if (isDrawing) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
  }
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
  ctx.closePath();
});

// Clear canvas
document.getElementById('clear-btn').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('prediction-result').innerText = '-';
});

// Predict digit
document.getElementById('predict-btn').addEventListener('click', async () => {
  // Preprocess the drawn image
  const image = preprocessImage(canvas);

  // Predict using the TensorFlow.js model
  const prediction = await model.predict(image).data();
  const predictedDigit = prediction.indexOf(Math.max(...prediction));

  // Display the result
  document.getElementById('prediction-result').innerText = predictedDigit;
});

// Preprocess the image (resize, normalize, etc.)
function preprocessImage(canvas) {
  // Create a temporary canvas to resize the image to 28x28 pixels
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  // Get the image data
  const imageData = tempCtx.getImageData(0, 0, 28, 28);

  // Convert to grayscale and normalize
  const input = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    const pixel = imageData.data[i];
    input.push(pixel / 255); // Normalize to [0, 1]
  }

  // Reshape to match the model's input shape
  const tensor = tf.tensor(input, [1, 28, 28, 1]);
  return tensor;
}