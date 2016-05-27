=============================================
Building a JIT: Per-function Lazy Compilation
=============================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 3 Introduction
======================

Welcome to Chapter 3 of the "Building an ORC-based JIT in LLVM" tutorial. This
chapter discusses lazy JITing and shows you how to enable it by adding an ORC
CompileOnDemand layer the JIT from `Chapter 2 <BuildingAJIT2.html>`_.

**To be done:**

**(1) Describe lazy function-at-a-time JITing and how it differs from the kind
of eager module-at-a-time JITing that we've been doing so far.**

**(2) Discuss CompileCallbackManagers and IndirectStubManagers.**

**(3) Describe CompileOnDemandLayer (automates these components and builds stubs
and lazy compilation callbacks for IR) and how to add it to the JIT.**

Full Code Listing
=================

Here is the complete code listing for our running example with a CompileOnDemand
layer added to enable lazy function-at-a-time compilation. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orc native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter3/KaleidoscopeJIT.h
   :language: c++

`Next: Extreme Laziness -- Using Compile Callbacks to JIT directly from ASTs <BuildingAJIT4.html>`_
