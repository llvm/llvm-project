# LLVM/SPIR-V Bi-Directional Translator

[![Build Status](https://travis-ci.org/KhronosGroup/SPIRV-LLVM-Translator.svg?branch=master)](https://travis-ci.org/KhronosGroup/SPIRV-LLVM-Translator)

This repository contains source code for the LLVM/SPIR-V Bi-Directional Translator, a library and tool for translation between LLVM IR and [SPIR-V](https://www.khronos.org/registry/spir-v/).

The LLVM/SPIR-V Bi-Directional Translator is open source software. You may freely distribute it under the terms of the license agreement found in LICENSE.txt.


## Directory Structure


The files/directories related to the translator:

* [include/LLVMSPIRVLib.h](include/LLVMSPIRVLib.h) - header file
* [lib/SPIRV](lib/SPIRV) - library for SPIR-V in-memory representation, decoder/encoder and LLVM/SPIR-V translator
* [tools/llvm-spirv](tools/llvm-spirv) - command line utility for translating between LLVM bitcode and SPIR-V binary

## Build Instructions

Master branch of this repo is aimed to be buildable with the latest LLVM version.

### Build with pre-installed LLVM

The translator can be built with the latest(nightly) package of LLVM. For Ubuntu and Debian systems LLVM provides repositories with nightly builds at http://apt.llvm.org/. For example the latest package for Ubuntu 16.04 can be installed with the following commands:
```
sudo add-apt-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main"
sudo apt-get update
sudo apt-get install llvm-9-dev llvm-9-tools libclang-9-dev
```
The installed version of LLVM will be used by default for out-of-tree build of the translator.
```
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build
cmake ..
make llvm-spirv -j`nproc`
```

### Build with pre-built LLVM

If you have a custom build(based on the latest version) of LLVM libraries you can link the translator against it. 
```
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build
cmake .. -DLLVM_DIR=<llvm_build_dir>/lib/cmake/llvm/
make llvm-spirv -j`nproc`
```
Where `llvm_build_dir` is the LLVM build directory.

### LLVM in-tree build

The translator can be built as a regular LLVM subproject. To do that you need to clone it to `llvm/projects` or `llvm/tools` directory. 
```
git clone http://llvm.org/git/llvm.git
cd llvm/projects
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
```
Run(re-run) cmake as usually for LLVM. After that you should have `llvm-spirv` and `check-llvm-spirv` targets available.
```
mkdir llvm/build && cd llvm/build 
cmake ..
make llvm-spirv -j`nproc`
```

## Test instructions

All tests related to the translator are placed in the [test](test) directory. Optionally the tests can make use of spirv-val (part of SPIRV-Tools) in order to validate the generated SPIR-V against the official SPIR-V specification.
In case tests are failing due to SPIRV-Tools not supporting certain SPIR-V features, please get an updated package. The `PKG_CONFIG_PATH` environmental variable can be used to let cmake point to a custom installation.

Execute the following command inside the build directory to run translator tests:
```
make test
```
This requires that the `-DLLVM_INCLUDE_TESTS=ON` argument was passed to CMake during the build step.

## Run Instructions for `llvm-spirv`


To translate between LLVM IR and SPIR-V:

1. Execute the following command to translate `input.bc` to `input.spv`
    ```
    llvm-spirv input.bc
    ```

2. Execute the following command to translate `input.spv` to `input.bc`
    ```
    llvm-spirv -r input.spv
    ```

3. Other options accepted by `llvm-spirv`

    * `-o file_name` - to specify output name
    * `-spirv-debug` - output debugging information
    * `-spirv-text` - read/write SPIR-V in an internal textual format for debugging purpose. The textual format is not defined by SPIR-V spec.
    * `-help` - to see full list of options

## Branching strategy

Code on the master branch in this repository is intended to be compatible with master/trunk branch of the [llvm](https://github.com/llvm-mirror/llvm) project. That is, for an OpenCL kernel compiled to llvm bitcode by the latest version(built with the latest git commit or svn revision) of Clang it should be possible to translate it to SPIR-V with the llvm-spirv tool.

All new development should be done on the master branch.

To have versions compatible with released versions of LLVM and Clang, corresponding branches are created in this repository. For example, to build translator with LLVM 7.0 ([release_70](https://github.com/llvm-mirror/llvm/tree/release_70)) one should use [llvm_release_70](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/llvm_release_70) branch.
