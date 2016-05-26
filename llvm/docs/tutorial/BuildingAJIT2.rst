====================================================================
Building a JIT: Adding Optimizations - An introduction to ORC Layers
====================================================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 2 Introduction
======================

Welcome to Chapter 2 of the "Building an ORC-based JIT in LLVM" tutorial. This
chapter shows you how to add IR optimization support to the KaleidoscopeJIT
class that was introduced in `Chapter 1 <BuildingAJIT1.html>`_ by adding a
new *ORC Layer* -- IRTransformLayer.

**To be done:**

**(1) Briefly describe FunctionPassManager and the optimizeModule
method (reference the Kaleidoscope language tutorial chapter 4 for more detail
about IR optimization - it's covered in detail there, here it just provides a
motivation for learning about layers).**

**(2) Describe IRTransformLayer, show how it is used to call our optimizeModule
method.**

**(3) Describe the ORC Layer concept using IRTransformLayer as an example.**

Full Code Listing
=================

Here is the complete code listing for our running example with an
IRTransformLayer added to enable optimization. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orc native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter2/KaleidoscopeJIT.h
   :language: c++

`Next: Adding Per-function Lazy Compilation <BuildingAJIT3.html>`_
