import nipype
from nipype.interfaces import afni
from nipype.interfaces import freesurfer
import os
import sys
import glob

path = sys.argv[1]
label_id = sys.argv[2]
output_name = sys.argv[3]

def extract_label(path, label_id, output_name):
    files2extract = glob.glob(path)
    for subj in range(len(files2extract)):
        print('extracting ',output_name,' for subject ', subj)
        subj_aparc = files2extract[subj]
        mc = freesurfer.MRIConvert()
        mc.inputs.in_file = subj_aparc
        mc.inputs.out_file = os.path.join(mc.inputs.in_file.split('.')[0] + mc.inputs.in_file.split('.')[1] + '.nii.gz')
        mc.run()

        calc = afni.Calc()
        calc.inputs.in_file_a = os.path.join(mc.inputs.in_file.split('.')[0] + mc.inputs.in_file.split('.')[1] + '.nii.gz')
        calc.inputs.expr = 'amongst(a,' + str(label_id) + ')'
        calc.inputs.out_file= os.path.join(os.path.dirname(mc.inputs.in_file) + '/' + output_name + '.nii.gz')
        calc.run()
extract_label(path, label_id, output_name)
