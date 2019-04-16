#!/usr/bin/env python

import os
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "spn", "src")

    sources_cpu = ['libspn_cpu.cpp', 'SoftProposalGenerator.cpp']
    sources_cuda = ['libspn_cuda.cpp', 'SoftProposalGenerator.cu']
    headers_cpu = ['libspn_cpu.h']
    headers_cuda = ['libspn_cuda.h']
    sources = sources_cpu
    headers = headers_cpu
    extension = CppExtension

    extra_objects = []
    define_macros = []
    with_cuda = False

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += sources_cuda
        headers += headers_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_objects += ['spn/src/libspn_kernel.cu.o']
        with_cuda = True


    sources = [os.path.join(extensions_dir, s) for s in sources]
    headers = [os.path.join(extensions_dir, h) for h in headers]
    include_dirs = [extensions_dir]

    for l in sources, headers, include_dirs:
        print(l)

    ext_modules = [
        extension(
            'spn._ext.libspn',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_objects=extra_objects
        )
    ]

    return ext_modules


setup(
    name="spn",
    version="1.0",
    description="Soft Proposal Networks, ICCV 2017",
    url="http://yzhu.work/spn",
    author="July",
    author_email="zhu.yee@outlook.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
