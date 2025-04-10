===================================================================
How to cross-compile Clang/LLVM using Clang/LLVM
===================================================================

Introduction
------------

This document contains information about building LLVM and
Clang on a host machine, targeting another platform.

For more information on how to use Clang as a cross-compiler,
please check https://clang.llvm.org/docs/CrossCompilation.html.

This document describes cross-building a compiler in a single stage, using an
existing ``clang`` install as the host compiler.

.. note::
  These instructions have been tested for targeting 32-bit ARM, AArch64, or
  64-bit RISC-V from an x86_64 Linux host. But should be equally applicable to
  any other target.

Setting up a sysroot
--------------------

You will need a sysroot that contains essential build dependencies compiled
for the target architecture. In this case, we will be using CMake and Ninja on
a Linux host and compiling against a Debian sysroot. Detailed instructions on
producing sysroots are outside of the scope of this documentation, but the
following instructions should work on any Linux distribution with these
pre-requisites:

 * ``binfmt_misc`` configured to execute ``qemu-user`` for binaries of the
   target architecture. This is done by installing the ``qemu-user-static``
   and ``binfmt-support`` packages on Debian-derived distributions.
 * Root access (setups involving ``proot`` or other tools to avoid this
   requirement may be possible, but aren't described here).
 * The ``debootstrap`` tool. This is available in most distributions.

The following snippet will initialise sysroots for 32-bit Arm, AArch64, and
64-bit RISC-V (just pick the target(s) you are interested in):

   .. code-block:: bash

    sudo debootstrap --arch=armhf --variant=minbase --include=build-essential,symlinks stable sysroot-deb-armhf-stable
    sudo debootstrap --arch=arm64 --variant=minbase --include=build-essential,symlinks stable sysroot-deb-arm64-stable
    sudo debootstrap --arch=riscv64 --variant=minbase --include=build-essential,symlinks unstable sysroot-deb-riscv64-unstable

The created sysroot may contain absolute symlinks, which will resolve to a
location within the host when accessed during compilation, so we must convert
any absolute symlinks to relative ones:

   .. code-block:: bash

    sudo chroot sysroot-of-your-choice symlinks -cr .


Configuring CMake and building
------------------------------

For more information on how to configure CMake for LLVM/Clang,
see :doc:`CMake`. Following CMake's recommended practice, we will create a
`toolchain file
<https://cmake.org/cmake/help/book/mastering-cmake/chapter/Cross%20Compiling%20With%20CMake.html#toolchain-files>`_. 

The following assumes you have a system install of ``clang`` and ``lld`` that
will be used for cross compiling and that the listed commands are executed
from within the root of a checkout of the ``llvm-project`` git repository.

First, set variables in your shell session that will be used throughout the
build instructions:

   .. code-block:: bash

    SYSROOT=$HOME/sysroot-deb-arm64-stable
    TARGET=aarch64-linux-gnu
    CFLAGS=""

To customise details of the compilation target or choose a different
architecture altogether, change the ``SYSROOT``,
``TARGET``, and ``CFLAGS`` variables to something matching your target. For
example, for 64-bit RISC-V you might set
``SYSROOT=$HOME/sysroot-deb-riscv64-unstable``, ``TARGET=riscv64-linux-gnu``
and ``CFLAGS="-march=rva20u64"``. Refer to documentation such as your target's
compiler documentation or processor manual for guidance on which ``CFLAGS``
settings may be appropriate. The specified ``TARGET`` should match the triple
used within the sysroot (i.e. ``$SYSROOT/usr/lib/$TARGET`` should exist).

Then execute the following snippet to create a toolchain file:

   .. code-block:: bash

    cat - <<EOF > $TARGET-clang.cmake
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSROOT "$SYSROOT")
    set(CMAKE_C_COMPILER_TARGET $TARGET)
    set(CMAKE_CXX_COMPILER_TARGET $TARGET)
    set(CMAKE_C_FLAGS_INIT "$CFLAGS")
    set(CMAKE_CXX_FLAGS_INIT "$CFLAGS")
    set(CMAKE_LINKER_TYPE LLD)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    EOF


Then configure and build by invoking ``cmake``:

   .. code-block:: bash

    cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="lld;clang" \
      -DCMAKE_TOOLCHAIN_FILE=$(pwd)/$TARGET-clang.cmake \
      -DLLVM_HOST_TRIPLE=$TARGET \
      -DCMAKE_INSTALL_PREFIX=$HOME/clang-$TARGET \
      -S llvm \
      -B build/$TARGET
    cmake --build build/$TARGET

These options from the toolchain file and ``cmake`` invocation above are
important:

 * ``CMAKE_SYSTEM_NAME``: Perhaps surprisingly, explicitly setting this
   variable `causes CMake to set
   CMAKE_CROSSCOMPIILING <https://cmake.org/cmake/help/latest/variable/CMAKE_CROSSCOMPILING.html#variable:CMAKE_CROSSCOMPILING>`_.
 * ``CMAKE_{C,CXX}_COMPILER_TARGET``: This will be used to set the
   ``--target`` argument to ``clang``. The triple should match the triple used
   within the sysroot (i.e. ``$SYSROOT/usr/lib/$TARGET`` should exist).
 * ``CMAKE_FIND_ROOT_PATH_MODE_*``: These `control the search behaviour for
   finding libraries, includes or binaries
   <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Cross%20Compiling%20With%20CMake.html#finding-external-libraries-programs-and-other-files>`_.
   Setting these prevents files for the host being used in the build.
 * ``LLVM_HOST_TRIPLE``: Specifies the target triple of the system the built
   LLVM will run on, which also implicitly sets other defaults such as
   ``LLVM_DEFAULT_TARGET_TRIPLE``. For example, if you are using an x86_64
   host to compile for RISC-V, this will be a RISC-V triple.
 * ``CMAKE_SYSROOT``: The path to the sysroot containing libraries and headers
   for the target.
 * ``CMAKE_INSTALL_PREFIX``: Setting this avoids installing binaries compiled
   for the target system into system directories for the host system. It is
   not required unless you are going to use the ``install`` target.

See `LLVM's build documentation
<https://llvm.org/docs/CMake.html#frequently-used-cmake-variables>`_ for more
guidance on CMake variables (e.g. ``LLVM_TARGETS_TO_BUILD`` may be useful if
your cross-compiled binaries only need to support compiling for one target).

Working around a ninja dependency issue
---------------------------------------

If you followed the instructions above to create a sysroot, you may run into a
`longstanding problem related to path canonicalization in ninja
<https://github.com/ninja-build/ninja/issues/1330>`_. GCC canonicalizes system
headers in dependency files, so when ninja reads them it does not need to do
so. Clang does not do this, and unfortunately ninja does not implement the
canonicalization logic at all, meaning for some system headers with symlinks
in the paths, it can incorrectly compute a non-existing path and consider it
as always modified.

If you are suffering from this issue, you will find any attempt at an
incremental build (including the suggested command to build the ``install``
target in the next section) results in recompiling everything.  ``ninja -C
build/$TARGET -t deps`` shows files in ``$SYSROOT/include/*`` that
do not exist (as the ``$SYSROOT/include`` folder does not exist) and you can
further confirm these files are causing ``ninja`` to determine a rebuild is
necessary with ``ninja -C build/$TARGET -d deps``.

A workaround is to create a symlink so that the incorrect
``$SYSROOT/include/*`` dependencies resolve to files within
``$SYSROOT/usr/include/*``. This works in practice for the simple
cross-compilation use case described here, but is not a general solution.

   .. code-block:: bash

    sudo ln -s usr/include $SYSROOT/include

Testing the just-built compiler
-------------------------------

Confirm the ``clang`` binary was built for the expected target architecture:

   .. code-block:: bash

    $ file -L ./build/aarch64-linux-gnu/bin/clang
    ./build/aarch64-linux-gnu/bin/clang: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 3.7.0, BuildID[sha1]=516b8b366a790fcd3563bee4aec0cdfcb90bb1c7, not stripped

If you have ``qemu-user`` installed you can test the produced target binary
either by invoking ``qemu-{target}-static`` directly:

   .. code-block:: bash

    $ qemu-aarch64-static -L $SYSROOT ./build/aarch64-linux-gnu/bin/clang --version
    clang version 21.0.0git (https://github.com/llvm/llvm-project cedfdc6e889c5c614a953ed1f44bcb45a405f8da)
    Target: aarch64-unknown-linux-gnu
    Thread model: posix
    InstalledDir: /home/asb/llvm-project/build/aarch64-linux-gnu/bin

Or, if binfmt_misc is configured (as was necessary for debootstrap):

   .. code-block:: bash

    $ export QEMU_LD_PREFIX=$SYSROOT; ./build/aarch64-linux-gnu/bin/clang --version
    clang version 21.0.0git (https://github.com/llvm/llvm-project cedfdc6e889c5c614a953ed1f44bcb45a405f8da)
    Target: aarch64-unknown-linux-gnu
    Thread model: posix
    InstalledDir: /home/asb/llvm-project/build/aarch64-linux-gnu/bin

Installing and using
--------------------

.. note::
  Use of the ``install`` target requires that you have set
  ``CMAKE_INSTALL_PREFIX`` otherwise it will attempt to install in
  directories under `/` on your host.

If you want to transfer a copy of the built compiler to another machine, you
can first install it to a location on the host via:

   .. code-block:: bash

    cmake --build build/$TARGET --target=install

This will install the LLVM/Clang headers, binaries, libraries, and other files
to paths within ``CMAKE_INSTALL_PREFIX``. Then tar that directory for transfer
to a device that runs the target architecture natively:

   .. code-block:: bash

    tar -czvf clang-$TARGET.tar.gz -C $HOME clang-$TARGET

The generated toolchain is portable, but requires compatible versions of any
shared libraries it links against. This means using a sysroot that is as
similar to your target operating system as possible is desirable. Other `CMake
variables <https://llvm.org/docs/CMake.html#frequently-used-cmake-variables>`_
may be helpful, for instance ``LLVM_STATIC_LINK_CXX_STDLIB``.
