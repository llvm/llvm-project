===================================================================
How to Cross Compile Compiler-rt Builtins For Arm
===================================================================

Introduction
============

This document contains information about building and testing the builtins part
of compiler-rt for an Arm target, from an x86_64 Linux machine.

While this document concentrates on Arm and Linux the general principles should
apply to other targets supported by compiler-rt. Further contributions for other
targets are welcome.

The instructions in this document depend on libraries and programs external to
LLVM, there are many ways to install and configure these dependencies so you
may need to adapt the instructions here to fit your own situation.

Prerequisites
=============

In this use case we will be using cmake on a Debian-based Linux system,
cross-compiling from an x86_64 host to a hard-float Armv7-A target. We will be
using as many of the LLVM tools as we can, but it is possible to use GNU
equivalents.

You will need:
 * A build of LLVM for the llvm-tools and LLVM CMake files.
 * A clang executable with support for the ``ARM`` target.
 * ``compiler-rt`` sources.
 * The ``qemu-arm`` user mode emulator.
 * An ``arm-linux-gnueabihf`` sysroot.

.. note::
  An existing sysroot is required because some of the builtins include C library
  headers and a sysroot is the easiest way to get those.

In this example we will be using ``ninja`` as the build tool.

See https://compiler-rt.llvm.org/ for information about the dependencies
on clang and LLVM.

See https://llvm.org/docs/GettingStarted.html for information about obtaining
the source for LLVM and compiler-rt.

``qemu-arm`` should be available as a package for your Linux distribution.

The most complicated of the prerequisites to satisfy is the ``arm-linux-gnueabihf``
sysroot. In theory it is possible to use the Linux distributions multiarch
support to fulfill the dependencies for building but unfortunately due to
``/usr/local/include`` being added some host includes are selected.

The easiest way to supply a sysroot is to download an ``arm-linux-gnueabihf``
toolchain from https://developer.arm.com/open-source/gnu-toolchain/gnu-a/downloads.

Building compiler-rt builtins for Arm
=====================================

We will be doing a standalone build of compiler-rt. The command is shown below.
Shell variables are used to simplify some of the options::

  LLVM_TOOLCHAIN=<path-to-llvm-install>/
  TARGET_TRIPLE=arm-none-linux-gnueabihf
  GCC_TOOLCHAIN=<path-to-gcc-toolchain>
  SYSROOT=${GCC_TOOLCHAIN}/${TARGET_TRIPLE}/libc
  COMPILE_FLAGS="-march=armv7-a"

  cmake ../llvm-project/compiler-rt \
    -G Ninja \
    -DCMAKE_AR=${LLVM_TOOLCHAIN}/bin/llvm-ar \
    -DCMAKE_NM=${LLVM_TOOLCHAIN}/bin/llvm-nm \
    -DCMAKE_RANLIB=${LLVM_TOOLCHAIN}/bin/llvm-ranlib \
    -DLLVM_CMAKE_DIR="${LLVM_TOOLCHAIN}/lib/cmake/llvm" \
    -DCMAKE_SYSROOT="${SYSROOT}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_ASM_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_TOOLCHAIN} \
    -DCMAKE_C_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCMAKE_C_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_TOOLCHAIN} \
    -DCMAKE_CXX_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCMAKE_CXX_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_BUILD_CRT=OFF \
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
    -DCOMPILER_RT_EMULATOR="qemu-arm -L ${SYSROOT}" \
    -DCOMPILER_RT_INCLUDE_TESTS=ON \
    -DCOMPILER_RT_TEST_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCOMPILER_RT_TEST_COMPILER_CFLAGS="--target=${TARGET_TRIPLE} ${COMPILE_FLAGS} --gcc-toolchain=${GCC_TOOLCHAIN} --sysroot=${SYSROOT} -fuse-ld=lld"

.. note::
  The command above also enables tests. Enabling tests is not required, more details
  in the testing section.

``CMAKE_<LANGUAGE>_<OPTION>`` options are set so that the correct ``--target``,
``--sysroot``, ``--gcc-toolchain`` and ``-march`` options will be given to the
compilers.

The combination of these settings needs to be enough to pass CMake's compiler
checks, compile compiler-rt and build the test cases.

The flags need to select:
 * The Arm target (``--target arm-none-linux-gnueabihf``)
 * The Arm architecture level (``-march=armv7-a``)
 * Whether to generate Arm (``-marm``, the default) or Thumb (``-mthumb``) instructions.

It is possible to pass all these flags to CMake using ``CMAKE_<LANGUAGE>_FLAGS``,
but the command above uses standard CMake options instead. If you need to
add flags that CMake cannot generate automatically, add them to
``CMAKE_<LANGUAGE>_FLAGS``.

When CMake has finished, build with Ninja::

  ninja builtins

Testing compiler-rt builtins using qemu-arm
===========================================

The following options are required to enable tests::

 -DCOMPILER_RT_EMULATOR="qemu-arm -L ${SYSROOT}" \
 -DCOMPILER_RT_INCLUDE_TESTS=ON \
 -DCOMPILER_RT_TEST_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
 -DCOMPILER_RT_TEST_COMPILER_CFLAGS="--target=${TARGET_TRIPLE} -march=armv7-a --gcc-toolchain=${GCC_TOOLCHAIN} --sysroot=${SYSROOT} -fuse-ld=lld"

This tells compiler-rt that we want to run tests on ``qemu-arm``. If you do not
want to run tests, remove these options from the CMake command.

Note that ``COMPILER_RT_TEST_COMPILER_CFLAGS`` contains the equivalent of the
options CMake generated for us with the first command. We must pass them
manually here because standard options like ``CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN``
do not apply here.

When CMake has finished, run the tests::

  ninja check-builtins

Troubleshooting
===============

The cmake try compile stage fails
---------------------------------
At an early stage cmake will attempt to compile and link a simple C program to
test if the toolchain is working.

This stage can often fail at link time if the ``--sysroot=``, ``--target`` or
``--gcc-toolchain=`` options are not passed to the compiler. Check the
``CMAKE_<LANGUAGE>_FLAGS`` and ``CMAKE_<LANGAUGE>_COMPILER_TARGET`` flags along
with any of the specific CMake sysroot and toolchain options.

It can be useful to build a simple example outside of cmake with your toolchain
to make sure it is working. For example::

  clang --target=arm-linux-gnueabi -march=armv7a --gcc-toolchain=/path/to/gcc-toolchain --sysroot=/path/to/gcc-toolchain/arm-linux-gnueabihf/libc helloworld.c

Clang uses the host header files
--------------------------------
On debian based systems it is possible to install multiarch support for
``arm-linux-gnueabi`` and ``arm-linux-gnueabihf``. In many cases clang can successfully
use this multiarch support when ``--gcc-toolchain=`` and ``--sysroot=`` are not supplied.
Unfortunately clang adds ``/usr/local/include`` before
``/usr/include/arm-linux-gnueabihf`` leading to errors when compiling the hosts
header files.

The multiarch support is not sufficient to build the builtins you will need to
use a separate ``arm-linux-gnueabihf`` toolchain.

No target passed to clang
-------------------------
If clang is not given a target it will typically use the host target, this will
not understand the Arm assembly language files resulting in error messages such
as ``error: unknown directive .syntax unified``.

You can check the clang invocation in the error message to see if there is no
``--target`` or if it is set incorrectly. The cause is usually
``CMAKE_ASM_FLAGS`` not containing ``--target`` or ``CMAKE_ASM_COMPILER_TARGET``
not being present.

Arm architecture not given
--------------------------
The ``--target=arm-linux-gnueabihf`` will default to Arm architecture v4t which
cannot assemble the barrier instructions used in the ``synch_and_fetch`` source
files.

The cause is usually a missing ``-march=armv7a`` from the ``CMAKE_ASM_FLAGS``.

Compiler-rt builds but the tests fail to build
----------------------------------------------
The flags used to build the tests are not the same as those used to build the
builtins. The c flags are provided by ``COMPILER_RT_TEST_COMPILE_CFLAGS`` and
the ``CMAKE_C_COMPILER_TARGET``, ``CMAKE_ASM_COMPILER_TARGET``,
``CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN`` and ``CMAKE_SYSROOT`` flags are not
applied to tests.

Make sure that ``COMPILER_RT_TEST_COMPILE_CFLAGS`` contains all the necessary
flags.


Modifications for other Targets
===============================

Arm Soft-Float Target
---------------------
The instructions for the Arm hard-float target can be used for the soft-float
target by substituting soft-float equivalents for the sysroot and target. The
target to use is:

* ``-DCMAKE_C_COMPILER_TARGET=arm-linux-gnueabi``

Depending on whether you want to use floating point instructions or not you
may need extra c-flags such as ``-mfloat-abi=softfp`` for use of floating-point
instructions, and ``-mfloat-abi=soft -mfpu=none`` for software floating-point
emulation.

You will need to use an ``arm-linux-gnueabi`` GNU toolchain for soft-float.

AArch64 Target
--------------
The instructions for Arm can be used for AArch64 by substituting AArch64
equivalents for the sysroot, emulator and target::

 -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu
 -DCOMPILER_RT_EMULATOR="qemu-aarch64 -L /path/to/aarch64/sysroot

You will also have to update any use of the target triple in compiler flags.
For instance in ``CMAKE_C_FLAGS`` and ``COMPILER_RT_TEST_COMPILER_CFLAGS``.

Armv6-m, Armv7-m and Armv7E-M targets
-------------------------------------
To build and test the libraries using a similar method to Armv7-A is possible
but more difficult. The main problems are:

* There is not a ``qemu-arm`` user-mode emulator for bare-metal systems.
  ``qemu-system-arm`` can be used but this is significantly more difficult
  to setup. This document does not explain how to do this.
* The targets to compile compiler-rt have the suffix ``-none-eabi``. This uses
  the BareMetal driver in clang and by default will not find the libraries
  needed to pass the cmake compiler check.

As the Armv6-M, Armv7-M and Armv7E-M builds of compiler-rt only use instructions
that are supported on Armv7-A we can still get most of the value of running the
tests using the same ``qemu-arm`` that we used for Armv7-A by building and
running the test cases for Armv7-A but using the builtins compiled for
Armv6-M, Armv7-M or Armv7E-M. This will test that the builtins can be linked
into a binary and execute the tests correctly but it will not catch if the
builtins use instructions that are supported on Armv7-A but not Armv6-M,
Armv7-M and Armv7E-M.

This requires a second ``arm-none-eabi`` toolchain for building the builtins.
Using a bare-metal toolchain ensures that the target and C library details are
specific to bare-metal instead of using Linux settings. This means that some
tests may behave differently compared to real hardware, but at least the content
of the builtins library is correct.

Below is an example that builds the builtins for Armv7-M, but runs the tests
as Armv7-A. It is presented in full, but is very similar to the earlier
command for Armv7-A build and test::

  LLVM_TOOLCHAIN=<path to llvm install>/

  # For the builtins.
  TARGET_TRIPLE=arm-none-eabi
  GCC_TOOLCHAIN=<path to arm-none-eabi toolchain>/
  SYSROOT=${GCC_TOOLCHAIN}/${TARGET_TRIPLE}/libc
  COMPILE_FLAGS="-march=armv7-m -mfpu=vfpv2"

  # For the test cases.
  A_PROFILE_TARGET_TRIPLE=arm-none-linux-gnueabihf
  A_PROFILE_GCC_TOOLCHAIN=<path to arm-none-linux-gnueabihf toolchain>/
  A_PROFILE_SYSROOT=${A_PROFILE_GCC_TOOLCHAIN}/${A_PROFILE_TARGET_TRIPLE}/libc

  cmake ../llvm-project/compiler-rt \
    -G Ninja \
    -DCMAKE_AR=${LLVM_TOOLCHAIN}/bin/llvm-ar \
    -DCMAKE_NM=${LLVM_TOOLCHAIN}/bin/llvm-nm \
    -DCMAKE_RANLIB=${LLVM_TOOLCHAIN}/bin/llvm-ranlib \
    -DLLVM_CMAKE_DIR="${LLVM_TOOLCHAIN}/lib/cmake/llvm" \
    -DCMAKE_SYSROOT="${SYSROOT}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_ASM_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_TOOLCHAIN} \
    -DCMAKE_C_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCMAKE_C_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET_TRIPLE}" \
    -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_TOOLCHAIN} \
    -DCMAKE_CXX_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCMAKE_CXX_FLAGS="${COMPILE_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_BUILD_CRT=OFF \
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
    -DCOMPILER_RT_EMULATOR="qemu-arm -L ${A_PROFILE_SYSROOT}" \
    -DCOMPILER_RT_INCLUDE_TESTS=ON \
    -DCOMPILER_RT_TEST_COMPILER=${LLVM_TOOLCHAIN}/bin/clang \
    -DCOMPILER_RT_TEST_COMPILER_CFLAGS="--target=${A_PROFILE_TARGET_TRIPLE} -march=armv7-a --gcc-toolchain=${A_PROFILE_GCC_TOOLCHAIN} --sysroot=${A_PROFILE_SYSROOT} -fuse-ld=lld" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCOMPILER_RT_OS_DIR="baremetal" \
    -DCOMPILER_RT_BAREMETAL_BUILD=ON

.. note::
  The sysroot used for compiling the tests is ``arm-linux-gnueabihf``, not
  ``arm-none-eabi`` which is used when compiling the builtins.

The Armv6-M builtins will use the soft-float ABI. When compiling the tests for
Armv7-A we must include ``"-mthumb -mfloat-abi=soft -mfpu=none"`` in the
test-c-flags. We must use an Armv7-A soft-float abi sysroot for ``qemu-arm``.

Depending on the linker used for the test cases you may encounter BuildAttribute
mismatches between the M-profile objects from compiler-rt and the A-profile
objects from the test. The lld linker does not check the profile
BuildAttribute so it can be used to link the tests by adding ``-fuse-ld=lld`` to the
``COMPILER_RT_TEST_COMPILER_CFLAGS``.
