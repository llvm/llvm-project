=============================================================================
Building a JIT: Remote-JITing -- Process Isolation and Laziness at a Distance
=============================================================================

.. contents::
   :local:

**This tutorial is under active development. It is incomplete and details may
change frequently.** Nonetheless we invite you to try it out as it stands, and
we welcome any feedback.

Chapter 5 Introduction
======================

Welcome to Chapter 5 of the "Building an ORC-based JIT in LLVM" tutorial. This
chapter introduces the ORC RemoteJIT Client/Server APIs and shows how to use
them to build a JIT stack that will execute its code via a communications
channel with a different process. This can be a separate process on the same
machine, a process on a different machine, or even a process on a different
platform/architecture. The code builds on top of the lazy-AST-compiling JIT
stack from `Chapter 4 <BuildingAJIT3.html>`_.

**To be done -- this is going to be a long one:**

**(1) Introduce channels, RPC, RemoteJIT Client and Server APIs**

**(2) Describe the client code in greater detail. Discuss modifications of the
KaleidoscopeJIT class, and the REPL itself.**

**(3) Describe the server code.**

**(4) Describe how to run the demo.**

Full Code Listing
=================

Here is the complete code listing for our running example that JITs lazily from
Kaleidoscope ASTS. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy
    clang++ -g Server/server.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` -O3 -o toy-server
    # Run
    ./toy-server &
    ./toy

Here is the code for the modified KaleidoscopeJIT:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter5/KaleidoscopeJIT.h
   :language: c++

And the code for the JIT server:

.. literalinclude:: ../../examples/Kaleidoscope/BuildingAJIT/Chapter5/Server/server.cpp
   :language: c++
