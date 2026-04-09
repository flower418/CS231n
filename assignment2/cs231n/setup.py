from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from pathlib import Path
import numpy

HERE = Path(__file__).resolve().parent

extensions = [
    Extension(
        "im2col_cython",
        [str(HERE / "im2col_cython.pyx")],
        include_dirs=[numpy.get_include()],
    ),
]

setup(ext_modules=cythonize(extensions),)
