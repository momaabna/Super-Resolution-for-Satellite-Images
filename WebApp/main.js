// Applying Super Resolution to an image based on ort model
const size = 64;
const scale = 4;
var file_width = 0;
var file_height = 0;
const num_threads = 8;
// Load the ORT model
var  modelPath ='models/RDN_model_64_X4_O9.onnx';
var options = {executionProviders: ['wasm']};

var progress_bar = document.getElementById("progress_bar");
var progress_text = document.getElementById("progress_text");
// after page  loaded complete

progress_bar.style.width = "0%";
progress_text.innerHTML = "0% loading model...";
var loadingModelPromise = ort.InferenceSession.create(modelPath,options).then((session) => {
    console.log('model loaded');
    progress_text.innerHTML = "100% model loaded";
    progress_bar.style.width = "100%";
    window.session = session;
    //window.session.num_threads = num_threads;
    //window.session.optimized = true;
    //use webgl backend
    //window.session.backend = 'wasm';
    window.session 
    //window.session.backend = 'wasm-simd';
    //window.session.backend = 'webgl-simd';
    //window.session.backend = 'wasm-threaded';
    //window.session.backend = 'webgl-threaded';




});

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
    }




// Preprocess the image input
function preprocess(imgData) {
    // Convert the image data to a tensor
    const width = size;
    const height = size;
    const preprocessedData = new Float32Array(width * height * 3);
    for (let i = 0; i < height * width; i++) {
        preprocessedData[i] = imgData[i * 4] / 255;
        preprocessedData[height * width + i] = imgData[i * 4 + 1] / 255;
        preprocessedData[height * width * 2 + i] = imgData[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', preprocessedData, [1, 3, height, width]);
}

// Postprocess the image output
function postprocess(outputTensor) {
    const width = size*scale;
    const height = size*scale;
    const outputData = outputTensor.data;
    const outputImageData = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; i++) {
        outputImageData[i * 4] = outputData[i] * 255;
        outputImageData[i * 4 + 1] = outputData[height * width + i] * 255;
        outputImageData[i * 4 + 2] = outputData[height * width * 2 + i] * 255;
        outputImageData[i * 4 + 3] = 255;
    }
    return outputImageData;

    
}

// Predict the image
async function predictImage() {
    progress_bar.style.width = "0%";
    progress_text.innerHTML = "0% preprocessing...";
    console.log('before  predictImage');
    // Load the image
    const image = document.getElementById('canvas');
    await loadingModelPromise;
    const width = size;
    const height = size;
    const outputCanvas = document.getElementById('output_canvas');
    const outputCtx = outputCanvas.getContext('2d');

    outputCanvas.width = file_width*scale
    outputCanvas.height = file_height*scale
    //sleep(1000);
    await new Promise(r => setTimeout(r, 1000));

    var number_of_chuncks_x = file_width/size;
    var number_of_chuncks_y = file_height/size;
    var number_of_chuncks = number_of_chuncks_x*number_of_chuncks_y;
    const ctx = document.getElementById('canvas').getContext('2d');
    const session = window.session;
    const input_name = session.inputNames[0];
    const output_name = session.outputNames[0];
    progress_bar.style.width = "0%";
    progress_text.innerHTML = "0% Processing...";
    //grid processing
    for (let i = 0; i < number_of_chuncks_x;i++){
        for (let j = 0; j < number_of_chuncks_y;j++){
            const imageData = ctx.getImageData(i*size, j*size, width, height);

            const inputTensor = preprocess(imageData.data);
            console.log('after  preprocess chunk '+i+' '+j);

            // Run the model with Tensor inputs and get the result
            console.log('before  run chunk '+i+' '+j);
        

            const input = { [input_name]: inputTensor}
            const output = await session.run(input, [output_name]);

            console.log('after  run chunk '+i+' '+j);
          
            const outputTensor = output[output_name];

            const outputImageData = postprocess(outputTensor);
    
            outputCtx.putImageData(new ImageData(outputImageData, width * scale, height * scale), i*size*scale, j*size*scale);
            console.log('after  predictImage chunk '+i+' '+j);
            progress_text.innerHTML = "Processing..."+Math.round((i*number_of_chuncks_y+j)/number_of_chuncks*100)+"%";
            progress_bar.style.width = ""+Math.round((i*number_of_chuncks_y+j)/number_of_chuncks*100)+"%";
            //sleep(1000);
            await new Promise(r => setTimeout(r, 1000));


        }
    }





}








// Load the image from file and pre-process it

var inputFile = document.getElementById("file");
inputFile.addEventListener("change", function () {
    if (this.files && this.files[0]) {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var img = new Image();
        img.src = URL.createObjectURL(this.files[0]);
        // get original file size and set canvas size
      

        img.onload = function () {
            console.log(this.width,this.height);
            //image padding

            canvas.width = this.width + (size - this.width % size);
            canvas.height = this.height + (size - this.height % size);
            file_width = this.width ;
            file_height = this.height ;
            ctx.drawImage(img, 0, 0, this.width, this.height);
        }

        
    }
}
);

document.getElementById("process").addEventListener("click",async function(){
    progress_bar.style.width = "0%";
    progress_text.innerHTML = "0% Start processing...";
    //sleep for 1 second
    await new Promise(r => setTimeout(r, 1000));
    await predictImage();
    progress_text.innerHTML = "100% Done";
    progress_bar.style.width = "100%";

}
);

document.getElementById("download").addEventListener("click", function(){
    var link = document.createElement('a');
    link.download = 'output.png';
    link.href = document.getElementById('output_canvas').toDataURL()
    link.click();
    link.delete;
}
);

//<!--model change-->
document.getElementById("model").addEventListener("change", function(){
    modelPath = this.value;
    console.log(modelPath);
    const mode = document.getElementById('mode').value;
    options = {executionProviders: [mode]};
    progress_bar.style.width = "0%";
    progress_text.innerHTML = "0% Loading model...";
    loadingModelPromise = ort.InferenceSession.create(modelPath, options).then(session => {
        window.session = session;

        console.log('after loading model');
        progress_text.innerHTML = "100% Model loaded.";
        progress_bar.style.width = "100%";
    });
    }
);

//<!--mode change-->
document.getElementById("mode").addEventListener("change", function(){
    mode = this.value;
    console.log(mode);
    options = {executionProviders: [mode]};
    progress_bar.style.width = "0%";
    progress_text.innerHTML = "0% Loading model...";
    loadingModelPromise =  ort.InferenceSession.create(modelPath, options).then(session => {
        window.session = session;
        console.log('after loading model');
        progress_text.innerHTML = "100% Model loaded.";
        progress_bar.style.width = "100%";
    });
    }
);
    
//<!--scale change-->
document.getElementById("scale").addEventListener("change", function(){
    scale = this.value;
    console.log(scale);
    }
);



//<!--tile size change-->
document.getElementById("tile_size").addEventListener("change", function(){
    size = this.value;
    console.log(size);
    }
);


