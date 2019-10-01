.. _index-subsystem-docs:

Subsystem Documentation
=======================

For API clients and LLVM developers.

.. toctree::
   :hidden:

   AliasAnalysis
   MemorySSA
   BitCodeFormat
   BlockFrequencyTerminology
   BranchWeightMetadata
   Bugpoint
   CodeGenerator
   ExceptionHandling
   AddingConstrainedIntrinsics
   LinkTimeOptimization
   SegmentedStacks
   TableGenFundamentals
   TableGen/index
   DebuggingJITedCode
   GoldPlugin
   MarkedUpDisassembly
   SystemLibrary
   SupportLibrary
   SourceLevelDebugging
   Vectorizers
   WritingAnLLVMBackend
   GarbageCollection
   WritingAnLLVMPass
   HowToUseAttributes
   StackMaps
   InAlloca
   BigEndianNEON
   CoverageMappingFormat
   Statepoints
   MergeFunctions
   TypeMetadata
   TransformMetadata
   FaultMaps
   Coroutines
   GlobalISel
   XRay
   XRayExample
   XRayFDRFormat
   PDB/index
   CFIVerify
   SpeculativeLoadHardening
   StackSafetyAnalysis
   LoopTerminology
   DependenceGraphs/index

:doc:`WritingAnLLVMPass`
   Information on how to write LLVM transformations and analyses.

:doc:`WritingAnLLVMBackend`
   Information on how to write LLVM backends for machine targets.

:doc:`CodeGenerator`
   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.

:doc:`TableGen <TableGen/index>`
   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.

:doc:`AliasAnalysis`
   Information on how to write a new alias analysis implementation or how to
   use existing analyses.

:doc:`MemorySSA`
   Information about the MemorySSA utility in LLVM, as well as how to use it.

:doc:`Source Level Debugging with LLVM <SourceLevelDebugging>`
   This document describes the design and philosophy behind the LLVM
   source-level debugger.

:doc:`Vectorizers`
   This document describes the current status of vectorization in LLVM.

:doc:`ExceptionHandling`
   This document describes the design and implementation of exception handling
   in LLVM.

:doc:`AddingConstrainedIntrinsics`
   Gives the steps necessary when adding a new constrained math intrinsic
   to LLVM.

:doc:`Bugpoint`
   Automatic bug finder and test-case reducer description and usage
   information.

:doc:`BitCodeFormat`
   This describes the file format and encoding used for LLVM "bc" files.

:doc:`Support Library <SupportLibrary>`
   This document describes the LLVM Support Library (``lib/Support``) and
   how to keep LLVM source code portable

:doc:`LinkTimeOptimization`
   This document describes the interface between LLVM intermodular optimizer
   and the linker and its design

:doc:`GoldPlugin`
   How to build your programs with link-time optimization on Linux.

:doc:`DebuggingJITedCode`
   How to debug JITed code with GDB.

:doc:`MCJITDesignAndImplementation`
   Describes the inner workings of MCJIT execution engine.

:doc:`ORCv2`
   Describes the design and implementation of the ORC APIs, including some
   usage examples, and a guide for users transitioning from ORCv1 to ORCv2.

:doc:`BranchWeightMetadata`
   Provides information about Branch Prediction Information.

:doc:`BlockFrequencyTerminology`
   Provides information about terminology used in the ``BlockFrequencyInfo``
   analysis pass.

:doc:`SegmentedStacks`
   This document describes segmented stacks and how they are used in LLVM.

:doc:`MarkedUpDisassembly`
   This document describes the optional rich disassembly output syntax.

:doc:`HowToUseAttributes`
  Answers some questions about the new Attributes infrastructure.

:doc:`StackMaps`
  LLVM support for mapping instruction addresses to the location of
  values and allowing code to be patched.

:doc:`BigEndianNEON`
  LLVM's support for generating NEON instructions on big endian ARM targets is
  somewhat nonintuitive. This document explains the implementation and rationale.

:doc:`CoverageMappingFormat`
  This describes the format and encoding used for LLVM’s code coverage mapping.

:doc:`MergeFunctions`
  Describes functions merging optimization.

:doc:`InAlloca`
  Description of the ``inalloca`` argument attribute.

:doc:`FaultMaps`
  LLVM support for folding control flow into faulting machine instructions.

:doc:`CompileCudaWithLLVM`
  LLVM support for CUDA.

:doc:`Coroutines`
  LLVM support for coroutines.

:doc:`GlobalISel`
  This describes the prototype instruction selection replacement, GlobalISel.

:doc:`XRay`
  High-level documentation of how to use XRay in LLVM.

:doc:`XRayExample`
  An example of how to debug an application with XRay.

:doc:`The Microsoft PDB File Format <PDB/index>`
  A detailed description of the Microsoft PDB (Program Database) file format.

:doc:`CFIVerify`
  A description of the verification tool for Control Flow Integrity.

:doc:`SpeculativeLoadHardening`
  A description of the Speculative Load Hardening mitigation for Spectre v1.

:doc:`StackSafetyAnalysis`
  This document describes the design of the stack safety analysis of local
  variables.

:doc:`LoopTerminology`
  A document describing Loops and associated terms as used in LLVM.

:doc:`Dependence Graphs <DependenceGraphs/index>`
  A description of the design of the various dependence graphs such as
  the DDG (Data Dependence Graph).

==================
Garbage Collection
==================

:doc:`GarbageCollection`
   The interfaces source-language compilers should use for compiling GC'd
   programs.

:doc:`Statepoints`
  This describes a set of experimental extensions for garbage
  collection support.
