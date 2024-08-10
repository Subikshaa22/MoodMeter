const express = require('express');
const fileUpload = require('express-fileupload');
const path = require('path');
const { exec } = require('child_process');

const app = express();
app.use(fileUpload());

// Serve static files from the "static" directory
app.use('/static', express.static(path.join(__dirname, 'static')));

app.set('view engine', 'ejs');

// Serve the index.html file on the root route
app.get('/', (req, res) => {
    res.render('index');
});

app.post('/predict', (req, res) => {
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }

    const audioFile = req.files.file;
    const uploadPath = path.join(__dirname, 'uploads', audioFile.name);

    // Save the uploaded file
    audioFile.mv(uploadPath, (err) => {
        if (err) {
            return res.status(500).send(err);
        }

        // Run the prediction script
        const command = `python predict.py ${uploadPath}`;
        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`exec error: ${error}`);
                return res.status(500).send('Prediction failed.');
            }

            // Send the prediction result back to the client
            res.render('index', { prediction: stdout });
        });
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
