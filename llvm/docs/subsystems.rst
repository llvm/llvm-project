.. _subsystems:

Subsystem Documentation
=======================

.. toctree::
   :hidden:

   AliasAnalysis
   BitCodeFormat
   BranchWeightMetadata
   Bugpoint
   CodeGenerator
   ExceptionHandling
   LinkTimeOptimization
   SegmentedStacks
   TableGenFundamentals
   DebuggingJITedCode
   GoldPlugin
   MarkedUpDisassembly

* `Writing an LLVM Pass <WritingAnLLVMPass.html>`_
    
   Information on how to write LLVM transformations and analyses.
    
* `Writing an LLVM Backend <WritingAnLLVMBackend.html>`_
    
   Information on how to write LLVM backends for machine targets.

* :ref:`code_generator`

   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.
    
* :ref:`tablegen`

   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.
    
* :ref:`alias_analysis`
    
   Information on how to write a new alias analysis implementation or how to
   use existing analyses.
    
* `Accurate Garbage Collection with LLVM <GarbageCollection.html>`_
    
   The interfaces source-language compilers should use for compiling GC'd
   programs.

* `Source Level Debugging with LLVM <SourceLevelDebugging.html>`_
    
   This document describes the design and philosophy behind the LLVM
   source-level debugger.
    
* :ref:`exception_handling`
    
   This document describes the design and implementation of exception handling
   in LLVM.
    
* :ref:`bugpoint`
    
   Automatic bug finder and test-case reducer description and usage
   information.
    
* :ref:`bitcode_format`
    
   This describes the file format and encoding used for LLVM "bc" files.
    
* `System Library <SystemLibrary.html>`_
    
   This document describes the LLVM System Library (<tt>lib/System</tt>) and
   how to keep LLVM source code portable
    
* :ref:`lto`
    
   This document describes the interface between LLVM intermodular optimizer
   and the linker and its design
    
* :ref:`gold-plugin`
    
   How to build your programs with link-time optimization on Linux.
    
* :ref:`debugging-jited-code`
    
   How to debug JITed code with GDB.
    
* :ref:`branch_weight`
    
   Provides information about Branch Prediction Information.

* :ref:`segmented_stacks`

   This document describes segmented stacks and how they are used in LLVM.

* `Howto: Implementing LLVM Integrated Assembler`_

   A simple guide for how to implement an LLVM integrated assembler for an
   architecture.

.. _`Howto: Implementing LLVM Integrated Assembler`: http://www.embecosm.com/download/ean10.html

* :ref:`marked_up_disassembly`

   This document describes the optional rich disassembly output syntax.

