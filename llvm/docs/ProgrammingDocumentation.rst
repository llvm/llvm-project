Programming Documentation
=========================

For developers of applications which use LLVM as a library.

.. toctree::
   :hidden:

   Atomics
   CommandLine
   CommandGuide/index
   ExtendingLLVM
   HowToSetUpLLVMStyleRTTI
   ProgrammersManual
   Extensions
   LibFuzzer
   FuzzingLLVM
   ScudoHardenedAllocator
   OptBisect
   GwpAsan

:doc:`Atomics`
  Information about LLVM's concurrency model.

:doc:`ProgrammersManual`
  Introduction to the general layout of the LLVM sourcebase, important classes
  and APIs, and some tips & tricks.

:doc:`Extensions`
  LLVM-specific extensions to tools and formats LLVM seeks compatibility with.

:doc:`HowToSetUpLLVMStyleRTTI`
  How to make ``isa<>``, ``dyn_cast<>``, etc. available for clients of your
  class hierarchy.

:doc:`ExtendingLLVM`
  Look here to see how to add instructions and intrinsics to LLVM.

:doc:`ScudoHardenedAllocator`
  A library that implements a security-hardened `malloc()`.

:doc:`GwpAsan`
  A sampled heap memory error detection toolkit designed for production use.

============
Command Line
============

:doc:`CommandLine`
  Provides information on using the command line parsing library.

:doc:`OptBisect`
  A command line option for debugging optimization-induced failures.

:doc:`LLVM Command Guide <CommandGuide/index>`
   A reference manual for the LLVM command line utilities ("man" pages for LLVM
   tools).

=========
LibFuzzer
=========

:doc:`LibFuzzer`
  A library for writing in-process guided fuzzers.

:doc:`FuzzingLLVM`
  Information on writing and using Fuzzers to find bugs in LLVM.