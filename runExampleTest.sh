#!/bin/bash
#Please download the checkpoint before running
python3 material_net.py --input_dir inputExamples/ --mode eval --output_dir examples_outputs --checkpoint . --imageFormat png --scale_size 256 --batch_size 1
