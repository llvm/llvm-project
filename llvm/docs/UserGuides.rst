User Guides
===========

NOTE: If you are a user who is only interested in using an LLVM-based compiler,
you should look into `Clang <https://clang.llvm.org>`_ instead. The
documentation here is intended for users who have a need to work with the
intermediate LLVM representation.

.. contents::
   :local:

.. toctree::
   :hidden:

   AddingConstrainedIntrinsics
   AdvancedBuilds
   AliasAnalysis
   AMDGPUUsage
   Benchmarking
   BigEndianNEON
   BuildingADistribution
   CFIVerify
   CMake
   CMakePrimer
   CodeGenerator
   CodeOfConduct
   CommandLine
   CompileCudaWithLLVM
   CoverageMappingFormat
   DebuggingJITedCode
   Docker
   ExtendingLLVM
   GoldPlugin
   HowToBuildOnARM
   HowToBuildWithPGO
   HowToCrossCompileBuiltinsOnArm
   HowToCrossCompileLLVM
   LinkTimeOptimization
   LoopTerminology
   MarkdownQuickstartTemplate
   MemorySSA
   MergeFunctions
   MCJITDesignAndImplementation
   NVPTXUsage
   Phabricator
   Passes
   ReportingGuide
   Remarks
   SourceLevelDebugging
   StackSafetyAnalysis
   SupportLibrary
   TableGen/index
   TableGenFundamentals
   Vectorizers
   WritingAnLLVMPass
   WritingAnLLVMBackend
   yaml2obj

Clang
-----

:doc:`HowToBuildOnARM`
   Notes on building and testing LLVM/Clang on ARM.

:doc:`HowToBuildWithPGO`
    Notes on building LLVM/Clang with PGO.

:doc:`HowToCrossCompileLLVM`
   Notes on cross-building and testing LLVM/Clang.

`How to build the C, C++, ObjC, and ObjC++ front end`__
   Instructions for building the clang front-end from source.

   .. __: https://clang.llvm.org/get_started.html

:doc:`CoverageMappingFormat`
  This describes the format and encoding used for LLVM’s code coverage mapping.

:doc:`CFIVerify`
  A description of the verification tool for Control Flow Integrity.

LLVM Builds and Distributions
-----------------------------

:doc:`BuildingADistribution`
  A best-practices guide for using LLVM's CMake build system to package and
  distribute LLVM-based tools.

:doc:`CMake`
   An addendum to the main Getting Started guide for those using the `CMake
   build system <http://www.cmake.org>`_.

:doc:`Docker`
   A reference for using Dockerfiles provided with LLVM.

:doc:`Support Library <SupportLibrary>`
   This document describes the LLVM Support Library (``lib/Support``) and
   how to keep LLVM source code portable

Optimizations
-------------

:doc:`WritingAnLLVMPass`
   Information on how to write LLVM transformations and analyses.

:doc:`Passes`
   A list of optimizations and analyses implemented in LLVM.

:doc:`StackSafetyAnalysis`
  This document describes the design of the stack safety analysis of local
  variables.

:doc:`MergeFunctions`
  Describes functions merging optimization.

:doc:`AliasAnalysis`
   Information on how to write a new alias analysis implementation or how to
   use existing analyses.

:doc:`MemorySSA`
   Information about the MemorySSA utility in LLVM, as well as how to use it.

:doc:`LoopTerminology`
  A document describing Loops and associated terms as used in LLVM.

:doc:`Vectorizers`
   This document describes the current status of vectorization in LLVM.

:doc:`LinkTimeOptimization`
   This document describes the interface between LLVM intermodular optimizer
   and the linker and its design

:doc:`GoldPlugin`
   How to build your programs with link-time optimization on Linux.

:doc:`Remarks`
   A reference on the implementation of remarks in LLVM.

:doc:`Source Level Debugging with LLVM <SourceLevelDebugging>`
   This document describes the design and philosophy behind the LLVM
   source-level debugger.

Code Generation
---------------

:doc:`WritingAnLLVMBackend`
   Information on how to write LLVM backends for machine targets.

:doc:`CodeGenerator`
   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.

:doc:`TableGen <TableGen/index>`
   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.

===
JIT
===

:doc:`MCJITDesignAndImplementation`
   Describes the inner workings of MCJIT execution engine.

:doc:`DebuggingJITedCode`
   How to debug JITed code with GDB.

Additional Topics
-----------------

:doc:`CommandLine`
  Provides information on using the command line parsing library.

:doc:`ExtendingLLVM`
  Look here to see how to add instructions and intrinsics to LLVM.

:doc:`AddingConstrainedIntrinsics`
   Gives the steps necessary when adding a new constrained math intrinsic
   to LLVM.

:doc:`HowToCrossCompileBuiltinsOnArm`
   Notes on cross-building and testing the compiler-rt builtins for Arm.

:doc:`BigEndianNEON`
  LLVM's support for generating NEON instructions on big endian ARM targets is
  somewhat nonintuitive. This document explains the implementation and rationale.

:doc:`CompileCudaWithLLVM`
  LLVM support for CUDA.

:doc:`NVPTXUsage`
   This document describes using the NVPTX backend to compile GPU kernels.

:doc:`AMDGPUUsage`
   This document describes using the AMDGPU backend to compile GPU kernels.