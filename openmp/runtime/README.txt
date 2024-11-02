
               README for the LLVM* OpenMP* Runtime Library
               ============================================

How to Build Documentation
==========================

The main documentation is in Doxygen* format, and this distribution
should come with pre-built PDF documentation in doc/Reference.pdf.
However, an HTML version can be built by executing:

% doxygen doc/doxygen/config

in the runtime directory.

That will produce HTML documentation in the doc/doxygen/generated
directory, which can be accessed by pointing a web browser at the
index.html file there.

If you don't have Doxygen installed, you can download it from
www.doxygen.org.


How to Build the LLVM* OpenMP* Runtime Library
==============================================
In-tree build:

$ cd where-you-want-to-live
Check out openmp into llvm/projects
$ cd where-you-want-to-build
$ mkdir build && cd build
$ cmake path/to/llvm -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make omp

Out-of-tree build:

$ cd where-you-want-to-live
Check out openmp
$ cd where-you-want-to-live/openmp/runtime
$ mkdir build && cd build
$ cmake path/to/openmp -DCMAKE_C_COMPILER=<C compiler> -DCMAKE_CXX_COMPILER=<C++ compiler>
$ make

For details about building, please look at README.rst in the parent directory.

Architectures Supported
=======================
* IA-32 architecture
* Intel(R) 64 architecture
* Intel(R) Many Integrated Core Architecture
* ARM* architecture
* Aarch64 (64-bit ARM) architecture
* IBM(R) Power architecture (big endian)
* IBM(R) Power architecture (little endian)
* MIPS and MIPS64 architecture
* RISCV64 architecture
* LoongArch64 architecture

Supported RTL Build Configurations
==================================

Supported Architectures: IA-32 architecture, Intel(R) 64, and
Intel(R) Many Integrated Core Architecture

              ----------------------------------------------
              |   icc/icl     |    gcc      |   clang      |
--------------|---------------|----------------------------|
| Linux* OS   |   Yes(1,5)    |  Yes(2,4)   | Yes(4,6,7)   |
| FreeBSD*    |   No          |  No         | Yes(4,6,7,8) |
| OS X*       |   Yes(1,3,4)  |  No         | Yes(4,6,7)   |
| Windows* OS |   Yes(1,4)    |  No         | No           |
------------------------------------------------------------

(1) On IA-32 architecture and Intel(R) 64, icc/icl versions 12.x are
    supported (12.1 is recommended).
(2) GCC* version 4.7 is supported.
(3) For icc on OS X*, OS X* version 10.5.8 is supported.
(4) Intel(R) Many Integrated Core Architecture not supported.
(5) On Intel(R) Many Integrated Core Architecture, icc/icl versions 13.0
    or later are required.
(6) Clang* version 3.3 is supported.
(7) Clang* currently does not offer a software-implemented 128 bit extended
    precision type.  Thus, all entry points reliant on this type are removed
    from the library and cannot be called in the user program.  The following
    functions are not available:
    __kmpc_atomic_cmplx16_*
    __kmpc_atomic_float16_*
    __kmpc_atomic_*_fp
(8) Community contribution provided AS IS, not tested by Intel.

Supported Architectures: IBM(R) Power 7 and Power 8

              -----------------------------
              |   gcc      |   clang      |
--------------|------------|--------------|
| Linux* OS   |  Yes(1,2)  | Yes(3,4)     |
-------------------------------------------

(1) On Power 7, gcc version 4.8.2 is supported.
(2) On Power 8, gcc version 4.8.2 is supported.
(3) On Power 7, clang version 3.7 is supported.
(4) On Power 8, clang version 3.7 is supported.


Front-end Compilers that work with this RTL
===========================================

The following compilers are known to do compatible code generation for
this RTL: clang (from the OpenMP development branch at
http://clang-omp.github.io/ ), Intel compilers, GCC.  See the documentation
for more details.

-----------------------------------------------------------------------

Notices
=======

*Other names and brands may be claimed as the property of others.
