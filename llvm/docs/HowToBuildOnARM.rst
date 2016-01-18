===================================================================
How To Build On ARM
===================================================================

Introduction
============

This document contains information about building/testing LLVM and
Clang on an ARM machine.

This document is *NOT* tailored to help you cross-compile LLVM/Clang
to ARM on another architecture, for example an x86_64 machine. To find
out more about cross-compiling, please check :doc:`HowToCrossCompileLLVM`.

Notes On Building LLVM/Clang on ARM
=====================================
Here are some notes on building/testing LLVM/Clang on ARM. Note that
ARM encompasses a wide variety of CPUs; this advice is primarily based
on the ARMv6 and ARMv7 architectures and may be inapplicable to older chips.

#. If you are building LLVM/Clang on an ARM board with 1G of memory or less,
   please use ``gold`` rather then GNU ``ld``.
   Building LLVM/Clang with ``--enable-optimized``
   is preferred since it consumes less memory. Otherwise, the building
   process will very likely fail due to insufficient memory. In any
   case it is probably a good idea to set up a swap partition.

#. If you want to run ``make check-all`` after building LLVM/Clang, to avoid
   false alarms (e.g., ARCMT failure) please use at least the following
   configuration:

   .. code-block:: bash

     $ ../$LLVM_SRC_DIR/configure --with-abi=aapcs-vfp

#. The most popular Linaro/Ubuntu OS's for ARM boards, e.g., the
   Pandaboard, have become hard-float platforms. The following set
   of configuration options appears to be a good choice for this
   platform:

   .. code-block:: bash

     ../$LLVM_SRC_DIR/configure --build=armv7l-unknown-linux-gnueabihf \
     --host=armv7l-unknown-linux-gnueabihf \
     --target=armv7l-unknown-linux-gnueabihf --with-cpu=cortex-a9 \
     --with-float=hard --with-abi=aapcs-vfp --with-fpu=neon \
     --enable-targets=arm --enable-optimized --enable-assertions

#. ARM development boards can be unstable and you may experience that cores
   are disappearing, caches being flushed on every big.LITTLE switch, and
   other similar issues.  To help ease the effect of this, set the Linux
   scheduler to "performance" on **all** cores using this little script:

   .. code-block:: bash

      # The code below requires the package 'cpufrequtils' to be installed.
      for ((cpu=0; cpu<`grep -c proc /proc/cpuinfo`; cpu++)); do
          sudo cpufreq-set -c $cpu -g performance
      done

#. Running the build on SD cards is ok, but they are more prone to failures
   than good quality USB sticks, and those are more prone to failures than
   external hard-drives (those are also a lot faster). So, at least, you
   should consider to buy a fast USB stick.  On systems with a fast eMMC,
   that's a good option too.

#. Make sure you have a decent power supply (dozens of dollars worth) that can
   provide *at least* 4 amperes, this is especially important if you use USB
   devices with your board.
