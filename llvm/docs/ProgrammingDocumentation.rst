Programming Documentation
=========================

For developers of applications which use LLVM as a library.

.. toctree::
   :hidden:

   Atomics
   CodingStandards
   CommandLine
   CompilerWriterInfo
   ExtendingLLVM
   HowToSetUpLLVMStyleRTTI
   ProgrammersManual
   Extensions
   LibFuzzer
   FuzzingLLVM
   ScudoHardenedAllocator
   OptBisect
   GwpAsan

:doc:`LLVM Language Reference Manual <LangRef>`
  Defines the LLVM intermediate representation and the assembly form of the
  different nodes.

:doc:`Atomics`
  Information about LLVM's concurrency model.

:doc:`ProgrammersManual`
  Introduction to the general layout of the LLVM sourcebase, important classes
  and APIs, and some tips & tricks.

:doc:`Extensions`
  LLVM-specific extensions to tools and formats LLVM seeks compatibility with.

:doc:`CommandLine`
  Provides information on using the command line parsing library.

:doc:`CodingStandards`
  Details the LLVM coding standards and provides useful information on writing
  efficient C++ code.

:doc:`HowToSetUpLLVMStyleRTTI`
  How to make ``isa<>``, ``dyn_cast<>``, etc. available for clients of your
  class hierarchy.

:doc:`ExtendingLLVM`
  Look here to see how to add instructions and intrinsics to LLVM.

`Doxygen generated documentation <http://llvm.org/doxygen/>`_
  (`classes <http://llvm.org/doxygen/inherits.html>`_)

`Documentation for Go bindings <http://godoc.org/llvm.org/llvm/bindings/go/llvm>`_

`Github Source Repository Browser <http://github.com/llvm/llvm-project//>`_
   ..

:doc:`CompilerWriterInfo`
  A list of helpful links for compiler writers.

:doc:`LibFuzzer`
  A library for writing in-process guided fuzzers.

:doc:`FuzzingLLVM`
  Information on writing and using Fuzzers to find bugs in LLVM.

:doc:`ScudoHardenedAllocator`
  A library that implements a security-hardened `malloc()`.

:doc:`OptBisect`
  A command line option for debugging optimization-induced failures.

:doc:`GwpAsan`
  A sampled heap memory error detection toolkit designed for production use.