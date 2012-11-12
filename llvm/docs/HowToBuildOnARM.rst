.. _how_to_build_on_arm:

===================================================================
How To Build On ARM
===================================================================

.. sectionauthor:: Wei-Ren Chen (陳韋任) <chenwj@iis.sinica.edu.tw>

Introduction
============

This document contains information about building/testing LLVM and
Clang on ARM.

Notes On Building LLVM/Clang on ARM
=====================================
Here are some notes on building/testing LLVM/Clang on ARM. Note that
ARM encompasses a wide variety of CPUs; this advice is primarily based
on the ARMv6 and ARMv7 architectures and may be inapplicable to older chips.

#. If you are building LLVM/Clang on an ARM board with 1G of memory or less,
   please use ``gold`` rather then GNU ``ld``.
   Building LLVM/Clang with ``--enable-optimized``
   is prefered since it consumes less memory. Otherwise, the building
   process will very likely fail due to insufficient memory. In any
   case it is probably a good idea to set up a swap partition.

#. If you want to run ``make
   check-all`` after building LLVM/Clang, to avoid false alarms (eg, ARCMT
   failure) please use at least the following configuration:

   .. code-block:: bash

     $ ../$LLVM_SRC_DIR/configure --with-abi=aapcs-vfp

#. The most popular linaro/ubuntu OS's for ARM boards, eg, the
   Pandaboard, have become hard-float platforms. The following set
   of configuration options appears to be a good choice for this
   platform:

   .. code-block:: bash

     ./configure --build=armv7l-unknown-linux-gnueabihf
     --host=armv7l-unknown-linux-gnueabihf
     --target=armv7l-unknown-linux-gnueabihf --with-cpu=cortex-a9
     --with-float=hard --with-abi=aapcs-vfp --with-fpu=neon
     --enable-targets=arm --disable-optimized --enable-assertions
