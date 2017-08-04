
    README for the LLVM* OpenMP* Offloading Runtime Library (libomptarget)
    ======================================================================

How to Build the LLVM* OpenMP* Offloading Runtime Library (libomptarget)
========================================================================
In-tree build:

$ cd where-you-want-to-live
Check out openmp (libomptarget lives under ./libomptarget) into llvm/projects
$ cd where-you-want-to-build
$ mkdir build && cd build
$ cmake path/to/llvm -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make omptarget

Out-of-tree build:

$ cd where-you-want-to-live
Check out openmp (libomptarget lives under ./libomptarget)
$ cd where-you-want-to-live/openmp/libomptarget
$ mkdir build && cd build
$ cmake path/to/openmp -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make

For details about building, please look at Build_With_CMake.txt

Architectures Supported
=======================
The current library has been only tested in Linux operating system and the 
following host architectures: 
* Intel(R) 64 architecture
* IBM(R) Power architecture (big endian)
* IBM(R) Power architecture (little endian)
* ARM(R) AArch64 architecture (little endian)

The currently supported offloading device architectures are:
* Intel(R) 64 architecture (generic 64-bit plugin - mostly for testing purposes)
* IBM(R) Power architecture (big endian) (generic 64-bit plugin - mostly for testing purposes)
* IBM(R) Power architecture (little endian) (generic 64-bit plugin - mostly for testing purposes)
* ARM(R) AArch64 architecture (little endian) (generic 64-bit plugin - mostly for testing purposes)
* CUDA(R) enabled 64-bit NVIDIA(R) GPU architectures

Supported RTL Build Configurations
==================================
Supported Architectures: Intel(R) 64, IBM(R) Power 7 and Power 8

              ---------------------------
              |   gcc      |   clang    |
--------------|------------|------------|
| Linux* OS   |  Yes(1)    |  Yes(2)    |
-----------------------------------------

(1) gcc version 4.8.2 or later is supported.
(2) clang version 3.7 or later is supported.


Front-end Compilers that work with this RTL
===========================================

The following compilers are known to do compatible code generation for
this RTL: 
  - clang (from https://github.com/clang-ykt )
  - clang (development branch at http://clang.llvm.org - several features still 
    under development)

-----------------------------------------------------------------------

Notices
=======
This library and related compiler support is still under development, so the 
employed interface is likely to change in the future.

*Other names and brands may be claimed as the property of others.

