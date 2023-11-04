========================================
How to Build the LLVM* OpenMP* Libraries
========================================
This repository requires `CMake <http://www.cmake.org/>`_ v2.8.0 or later.  LLVM
and Clang need a more recent version which also applies for in-tree builds.  For
more information than available in this document please see
`LLVM's CMake documentation <https://llvm.org/docs/CMake.html>`_ and the
`official documentation <https://cmake.org/cmake/help/v2.8.0/cmake.html>`_.

.. contents::
   :local:

How to Call CMake Initially, then Repeatedly
============================================
- When calling CMake for the first time, all needed compiler options must be
  specified on the command line.  After this initial call to CMake, the compiler
  definitions must not be included for further calls to CMake.  Other options
  can be specified on the command line multiple times including all definitions
  in the build options section below.
- Example of configuring, building, reconfiguring, rebuilding:

  .. code-block:: console

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..  # Initial configuration
    $ make
    ...
    $ make clean
    $ cmake -DCMAKE_BUILD_TYPE=Debug ..                               # Second configuration
    $ make
    ...
    $ rm -rf *
    $ cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..        # Third configuration
    $ make

- Notice in the example how the compiler definitions are only specified for an
  empty build directory, but other build options are used at any time.
- The file ``CMakeCache.txt`` which is created after the first call to CMake is
  a configuration file which holds all values for the build options.  These
  values can be changed using a text editor to modify ``CMakeCache.txt`` as
  opposed to using definitions on the command line.
- To have CMake create a particular type of build generator file simply include
  the ``-G <Generator name>`` option:

  .. code-block:: console

    $ cmake -G "Unix Makefiles" ...

  You can see a list of generators CMake supports by executing the cmake command
  with no arguments.

Instructions to Build
=====================
.. code-block:: console

 $ cd openmp_top_level/ [ this directory with libomptarget/, runtime/, etc. ]
 $ mkdir build
 $ cd build

 [ Unix* Libraries ]
 $ cmake -DCMAKE_C_COMPILER=<C Compiler> -DCMAKE_CXX_COMPILER=<C++ Compiler> ..

 [ Windows* Libraries ]
 $ cmake -G <Generator Type> -DCMAKE_C_COMPILER=<C Compiler> -DCMAKE_CXX_COMPILER=<C++ Compiler> -DCMAKE_ASM_MASM_COMPILER=[ml | ml64] -DCMAKE_BUILD_TYPE=Release ..

 $ make
 $ make install

CMake Options
=============
Builds with CMake can be customized by means of options as already seen above.
One possibility is to pass them via the command line:

.. code-block:: console

  $ cmake -DOPTION=<value> path/to/source

.. note:: The first value listed is the respective default for that option.

Generic Options
---------------
For full documentation consult the CMake manual or execute
``cmake --help-variable VARIABLE_NAME`` to get information about a specific
variable.

**CMAKE_BUILD_TYPE** = ``Release|Debug|RelWithDebInfo``
  Build type can be ``Release``, ``Debug``, or ``RelWithDebInfo`` which chooses
  the optimization level and presence of debugging symbols.

**CMAKE_C_COMPILER** = <C compiler name>
  Specify the C compiler.

**CMAKE_CXX_COMPILER** = <C++ compiler name>
  Specify the C++ compiler.

**CMAKE_Fortran_COMPILER** = <Fortran compiler name>
  Specify the Fortran compiler. This option is only needed when
  **LIBOMP_FORTRAN_MODULES** is ``ON`` (see below).  So typically, a Fortran
  compiler is not needed during the build.

**CMAKE_ASM_MASM_COMPILER** = ``ml|ml64``
  This option is only relevant for Windows*.

Options for all Libraries
-------------------------

**OPENMP_ENABLE_WERROR** = ``OFF|ON``
  Treat warnings as errors and fail, if a compiler warning is triggered.

**OPENMP_LIBDIR_SUFFIX** = ``""``
  Extra suffix to append to the directory where libraries are to be installed.

**OPENMP_TEST_C_COMPILER** = ``${CMAKE_C_COMPILER}``
  Compiler to use for testing. Defaults to the compiler that was also used for
  building.

**OPENMP_TEST_CXX_COMPILER** = ``${CMAKE_CXX_COMPILER}``
  Compiler to use for testing. Defaults to the compiler that was also used for
  building.

**OPENMP_TEST_Fortran_COMPILER** = ``${CMAKE_Fortran_COMPILER}``
  Compiler to use for testing. Defaults to the compiler that was also used for
  building. Will default to flang if build is in-tree.

**OPENMP_LLVM_TOOLS_DIR** = ``/path/to/built/llvm/tools``
  Additional path to search for LLVM tools needed by tests.

**OPENMP_LLVM_LIT_EXECUTABLE** = ``/path/to/llvm-lit``
  Specify full path to ``llvm-lit`` executable for running tests.  The default
  is to search the ``PATH`` and the directory in **OPENMP_LLVM_TOOLS_DIR**.

**OPENMP_FILECHECK_EXECUTABLE** = ``/path/to/FileCheck``
  Specify full path to ``FileCheck`` executable for running tests.  The default
  is to search the ``PATH`` and the directory in **OPENMP_LLVM_TOOLS_DIR**.

**OPENMP_NOT_EXECUTABLE** = ``/path/to/not``
  Specify full path to ``not`` executable for running tests.  The default
  is to search the ``PATH`` and the directory in **OPENMP_LLVM_TOOLS_DIR**.

Options for ``libomp``
----------------------

**LIBOMP_ARCH** = ``aarch64|arm|i386|loongarch64|mic|mips|mips64|ppc64|ppc64le|x86_64|riscv64|s390x``
  The default value for this option is chosen based on probing the compiler for
  architecture macros (e.g., is ``__x86_64__`` predefined by compiler?).

**LIBOMP_MIC_ARCH** = ``knc|knf``
  Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture) to
  build for.  This value is ignored if **LIBOMP_ARCH** does not equal ``mic``.

**LIBOMP_LIB_TYPE** = ``normal|profile|stubs``
  Library type can be ``normal``, ``profile``, or ``stubs``.

**LIBOMP_USE_VERSION_SYMBOLS** = ``ON|OFF``
  Use versioned symbols for building the library.  This option only makes sense
  for ELF based libraries where version symbols are supported (Linux*, some BSD*
  variants).  It is ``OFF`` by default for Windows* and macOS*, but ``ON`` for
  other Unix based operating systems.

**LIBOMP_ENABLE_SHARED** = ``ON|OFF``
  Build a shared library.  If this option is ``OFF``, static OpenMP libraries
  will be built instead of dynamic ones.

  .. note::

    Static libraries are not supported on Windows*.

**LIBOMP_FORTRAN_MODULES** = ``OFF|ON``
  Create the Fortran modules (requires Fortran compiler).

macOS* Fat Libraries
""""""""""""""""""""
On macOS* machines, it is possible to build universal (or fat) libraries which
include both i386 and x86_64 architecture objects in a single archive.

.. code-block:: console

  $ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_OSX_ARCHITECTURES='i386;x86_64' ..
  $ make

There is also an option **LIBOMP_OSX_ARCHITECTURES** which can be set in case
this is an LLVM source tree build. It will only apply for the ``libomp`` library
avoids having the entire LLVM/Clang build produce universal binaries.

Optional Features
"""""""""""""""""

**LIBOMP_USE_ADAPTIVE_LOCKS** = ``ON|OFF``
  Include adaptive locks, based on Intel(R) Transactional Synchronization
  Extensions (Intel(R) TSX).  This feature is x86 specific and turned ``ON``
  by default for IA-32 architecture and Intel(R) 64 architecture.

**LIBOMP_USE_INTERNODE_ALIGNMENT** = ``OFF|ON``
  Align certain data structures on 4096-byte.  This option is useful on
  multi-node systems where a small ``CACHE_LINE`` setting leads to false sharing.

**LIBOMP_OMPT_SUPPORT** = ``ON|OFF``
  Include support for the OpenMP Tools Interface (OMPT).
  This option is supported and ``ON`` by default for x86, x86_64, AArch64,
  PPC64, RISCV64, LoongArch64, and s390x on Linux* and macOS*.
  This option is ``OFF`` if this feature is not supported for the platform.

**LIBOMP_OMPT_OPTIONAL** = ``ON|OFF``
  Include support for optional OMPT functionality.  This option is ignored if
  **LIBOMP_OMPT_SUPPORT** is ``OFF``.

**LIBOMP_STATS** = ``OFF|ON``
  Include stats-gathering code.

**LIBOMP_USE_DEBUGGER** = ``OFF|ON``
  Include the friendly debugger interface.

**LIBOMP_USE_HWLOC** = ``OFF|ON``
  Use `OpenMPI's hwloc library <https://www.open-mpi.org/projects/hwloc/>`_ for
  topology detection and affinity.

**LIBOMP_HWLOC_INSTALL_DIR** = ``/path/to/hwloc/install/dir``
  Specify install location of hwloc.  The configuration system will look for
  ``hwloc.h`` in ``${LIBOMP_HWLOC_INSTALL_DIR}/include`` and the library in
  ``${LIBOMP_HWLOC_INSTALL_DIR}/lib``.  The default is ``/usr/local``.
  This option is only used if **LIBOMP_USE_HWLOC** is ``ON``.

Additional Compiler Flags
"""""""""""""""""""""""""

These flags are **appended**, they do not overwrite any of the preset flags.

**LIBOMP_CPPFLAGS** = <space-separated flags>
  Additional C preprocessor flags.

**LIBOMP_CXXFLAGS** = <space-separated flags>
  Additional C++ compiler flags.

**LIBOMP_ASMFLAGS** = <space-separated flags>
  Additional assembler flags.

**LIBOMP_LDFLAGS** = <space-separated flags>
  Additional linker flags.

**LIBOMP_LIBFLAGS** = <space-separated flags>
  Additional libraries to link.

**LIBOMP_FFLAGS** = <space-separated flags>
  Additional Fortran compiler flags.

Options for ``libomptarget``
----------------------------

An installed LLVM package is a prerequisite for building ``libomptarget``
library. So ``libomptarget`` may only be built in two cases:

- As a project of a regular LLVM build via **LLVM_ENABLE_PROJECTS**,
  **LLVM_EXTERNAL_PROJECTS**, or **LLVM_ENABLE_RUNTIMES** or
- as a standalone project build that uses a pre-installed LLVM package.
  In this mode one has to make sure that the default CMake
  ``find_package(LLVM)`` call `succeeds <https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure>`_.

**LIBOMPTARGET_OPENMP_HEADER_FOLDER** = ``""``
  Path of the folder that contains ``omp.h``.  This is required for testing
  out-of-tree builds.

**LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER** = ``""``
  Path of the folder that contains ``libomp.so``, and ``libLLVMSupport.so``
  when profiling is enabled.  This is required for testing.

Options for ``NVPTX device RTL``
--------------------------------

**LIBOMPTARGET_NVPTX_ENABLE_BCLIB** = ``ON|OFF``
  Enable CUDA LLVM bitcode offloading device RTL. This is used for link time
  optimization of the OMP runtime and application code. This option is enabled
  by default if the build system determines that `CMAKE_C_COMPILER` is able to
  compile and link the library.

**LIBOMPTARGET_NVPTX_CUDA_COMPILER** = ``""``
  Location of a CUDA compiler capable of emitting LLVM bitcode. Currently only
  the Clang compiler is supported. This is only used when building the CUDA LLVM
  bitcode offloading device RTL. If unspecified, either the Clang from the build
  itself is used (i.e. an in-tree build with LLVM_ENABLE_PROJECTS including
  clang), or the Clang compiler that the build uses as C compiler
  (CMAKE_C_COMPILER; only if it is Clang). The latter is common for a
  stage2-build or when using -DLLVM_ENABLE_RUNTIMES=openmp.

**LIBOMPTARGET_NVPTX_BC_LINKER** = ``""``
  Location of a linker capable of linking LLVM bitcode objects. This is only
  used when building the CUDA LLVM bitcode offloading device RTL. If
  unspecified, either the llvm-link in that same directory as
  LIBOMPTARGET_NVPTX_CUDA_COMPILER is used, or the llvm-link from the
  same build (available in an in-tree build).

**LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER** = ``""``
  Host compiler to use with NVCC. This compiler is not going to be used to
  produce any binary. Instead, this is used to overcome the input compiler
  checks done by NVCC. E.g. if using a default host compiler that is not
  compatible with NVCC, this option can be use to pass to NVCC a valid compiler
  to avoid the error.

 **LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES** = ``35``
  List of CUDA compute capabilities that should be supported by the NVPTX
  device RTL. E.g. for compute capabilities 6.0 and 7.0, the option "60;70"
  should be used. Compute capability 3.5 is the minimum required.

 **LIBOMPTARGET_NVPTX_DEBUG** = ``OFF|ON``
  Enable printing of debug messages from the NVPTX device RTL.

**LIBOMPTARGET_LIT_ARGS** = ``""``
  Arguments given to lit. ``make check-libomptarget`` and
  ``make check-libomptarget-*`` are affected. For example, use
  ``LIBOMPTARGET_LIT_ARGS="-j4"`` to force ``lit`` to start only four parallel
  jobs instead of by default the number of threads in the system.

Example Usages of CMake
=======================

Typical Invocations
-------------------

.. code-block:: console

  $ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
  $ cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
  $ cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc ..

Advanced Builds with Various Options
------------------------------------

- Build the i386 Linux* library using GCC*

  .. code-block:: console

    $ cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLIBOMP_ARCH=i386 ..

- Build the x86_64 debug Mac library using Clang*

  .. code-block:: console

    $ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLIBOMP_ARCH=x86_64 -DCMAKE_BUILD_TYPE=Debug ..

- Build the library (architecture determined by probing compiler) using the
  Intel(R) C Compiler and the Intel(R) C++ Compiler.  Also, create Fortran
  modules with the Intel(R) Fortran Compiler.

  .. code-block:: console

    $ cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DLIBOMP_FORTRAN_MODULES=on ..

- Have CMake find the C/C++ compiler and specify additional flags for the
  preprocessor and C++ compiler.

  .. code-blocks:: console

    $ cmake -DLIBOMP_CPPFLAGS='-DNEW_FEATURE=1 -DOLD_FEATURE=0' -DLIBOMP_CXXFLAGS='--one-specific-flag --two-specific-flag' ..

- Build the stubs library

  .. code-blocks:: console

    $ cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLIBOMP_LIB_TYPE=stubs ..

**Footnotes**

.. [*] Other names and brands may be claimed as the property of others.
