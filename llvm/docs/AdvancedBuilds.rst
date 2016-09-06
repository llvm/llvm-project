=============================
Advanced Build Configurations
=============================

.. contents::
   :local:

Introduction
============

`CMake <http://www.cmake.org/>`_ is a cross-platform build-generator tool. CMake
does not build the project, it generates the files needed by your build tool
(GNU make, Visual Studio, etc.) for building LLVM.

If **you are a new contributor**, please start with the :doc:`GettingStarted` or
:doc:`CMake` pages. This page is intended for users doing more complex builds.

Many of the examples below are written assuming specific CMake Generators.
Unless otherwise explicitly called out these commands should work with any CMake
generator.

Bootstrap Builds
================

The Clang CMake build system supports bootstrap (aka multi-stage) builds. At a
high level a multi-stage build is a chain of builds that pass data from one
stage into the next. The most common and simple version of this is a traditional
bootstrap build.

In a simple two-stage bootstrap build, we build clang using the system compiler,
then use that just-built clang to build clang again. In CMake this simplest form
of a bootstrap build can be configured with a single option,
CLANG_ENABLE_BOOTSTRAP.

.. code-block:: console

  $ cmake -G Ninja -DCLANG_ENABLE_BOOTSTRAP=On <path to source>
  $ ninja stage2

This command itself isn't terribly useful because it assumes default
configurations for each stage. The next series of examples utilize CMake cache
scripts to provide more complex options.

The clang build system refers to builds as stages. A stage1 build is a standard
build using the compiler installed on the host, and a stage2 build is built
using the stage1 compiler. This nomenclature holds up to more stages too. In
general a stage*n* build is built using the output from stage*n-1*.

Apple Clang Builds (A More Complex Bootstrap)
=============================================

Apple's Clang builds are a slightly more complicated example of the simple
bootstrapping scenario. Apple Clang is built using a 2-stage build.

The stage1 compiler is a host-only compiler with some options set. The stage1
compiler is a balance of optimization vs build time because it is a throwaway.
The stage2 compiler is the fully optimized compiler intended to ship to users.

Setting up these compilers requires a lot of options. To simplify the
configuration the Apple Clang build settings are contained in CMake Cache files.
You can build an Apple Clang compiler using the following commands:

.. code-block:: console

  $ cmake -G Ninja -C <path to clang>/cmake/caches/Apple-stage1.cmake <path to source>
  $ ninja stage2-distribution

This CMake invocation configures the stage1 host compiler, and sets
CLANG_BOOTSTRAP_CMAKE_ARGS to pass the Apple-stage2.cmake cache script to the
stage2 configuration step.

When you build the stage2-distribution target it builds the minimal stage1
compiler and required tools, then configures and builds the stage2 compiler
based on the settings in Apple-stage2.cmake.

This pattern of using cache scripts to set complex settings, and specifically to
make later stage builds include cache scripts is common in our more advanced
build configurations.

Multi-stage PGO
===============

Profile-Guided Optimizations (PGO) is a really great way to optimize the code
clang generates. Our multi-stage PGO builds are a workflow for generating PGO
profiles that can be used to optimize clang.

At a high level, the way PGO works is that you build an instrumented compiler,
then you run the instrumented compiler against sample source files. While the
instrumented compiler runs it will output a bunch of files containing
performance counters (.profraw files). After generating all the profraw files
you use llvm-profdata to merge the files into a single profdata file that you
can feed into the LLVM_PROFDATA_FILE option.

Our PGO.cmake cache script automates that whole process. You can use it by
running:

.. code-block:: console

  $ cmake -G Ninja -C <path_to_clang>/cmake/caches/PGO.cmake <source dir>
  $ ninja stage2-instrumented-generate-profdata

If you let that run for a few hours or so, it will place a profdata file in your
build directory. This takes a really long time because it builds clang twice,
and you *must* have compiler-rt in your build tree.

This process uses any source files under the perf-training directory as training
data as long as the source files are marked up with LIT-style RUN lines.

After it finishes you can use “find . -name clang.profdata” to find it, but it
should be at a path something like:

.. code-block:: console

  <build dir>/tools/clang/stage2-instrumented-bins/utils/perf-training/clang.profdata

You can feed that file into the LLVM_PROFDATA_FILE option when you build your
optimized compiler.

The PGO came cache has a slightly different stage naming scheme than other
multi-stage builds. It generates three stages; stage1, stage2-instrumented, and
stage2. Both of the stage2 builds are built using the stage1 compiler.

The PGO came cache generates the following additional targets:

**stage2-instrumented**
  Builds a stage1 x86 compiler, runtime, and required tools (llvm-config,
  llvm-profdata) then uses that compiler to build an instrumented stage2 compiler.

**stage2-instrumented-generate-profdata**
  Depends on "stage2-instrumented" and will use the instrumented compiler to
  generate profdata based on the training files in <clang>/utils/perf-training

**stage2**
  Depends of "stage2-instrumented-generate-profdata" and will use the stage1
  compiler with the stage2 profdata to build a PGO-optimized compiler.

**stage2-check-llvm**
  Depends on stage2 and runs check-llvm using the stage2 compiler.

**stage2-check-clang**
  Depends on stage2 and runs check-clang using the stage2 compiler.

**stage2-check-all**
  Depends on stage2 and runs check-all using the stage2 compiler.

**stage2-test-suite**
  Depends on stage2 and runs the test-suite using the stage3 compiler (requires
  in-tree test-suite).

3-Stage Non-Determinism
=======================

In the ancient lore of compilers non-determinism is like the multi-headed hydra.
Whenever it's head pops up, terror and chaos ensue.

Historically one of the tests to verify that a compiler was deterministic would
be a three stage build. The idea of a three stage build is you take your sources
and build a compiler (stage1), then use that compiler to rebuild the sources
(stage2), then you use that compiler to rebuild the sources a third time
(stage3) with an identical configuration to the stage2 build. At the end of
this, you have a stage2 and stage3 compiler that should be bit-for-bit
identical.

You can perform one of these 3-stage builds with LLVM & clang using the
following commands:

.. code-block:: console

  $ cmake -G Ninja -C <path_to_clang>/cmake/caches/3-stage.cmake <source dir>
  $ ninja stage3

After the build you can compare the stage2 & stage3 compilers. We have a bot
setup `here <http://lab.llvm.org:8011/builders/clang-3stage-ubuntu>`_ that runs
this build and compare configuration.
