#!/bin/bash

python3 material_net.py --mode train --output_dir $outputDir --input_dir $inputDir/trainBlended --batch_size 8 --loss render --useLog
