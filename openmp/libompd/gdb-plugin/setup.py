from setuptools import setup, Extension, find_packages

import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
omp_include_dir = os.environ.get('LIBOMP_INCLUDE_DIR', dir_path)
python_include_dir = os.environ.get('PYTHON_HEADERS', dir_path)

print("find_packages : ", find_packages())
setup(
    name='ompd',
    version='1.0',
    py_modules=['loadompd'],
    setup_requires=['wheel'],
    packages=find_packages(),
	ext_modules=[Extension('ompd.ompdModule', [dir_path+'/ompdModule.c', dir_path+'/ompdAPITests.c'], include_dirs=[omp_include_dir])]
)
