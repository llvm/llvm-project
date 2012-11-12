.. _programming:

Programming Documentation
=========================

.. toctree::
   :hidden:

   Atomics
   CodingStandards
   CommandLine
   CompilerWriterInfo
   ExtendingLLVM
   HowToSetUpLLVMStyleRTTI

* `LLVM Language Reference Manual <LangRef.html>`_

  Defines the LLVM intermediate representation and the assembly form of the
  different nodes.

* :ref:`atomics`

  Information about LLVM's concurrency model.

* `The LLVM Programmers Manual <ProgrammersManual.html>`_

  Introduction to the general layout of the LLVM sourcebase, important classes
  and APIs, and some tips & tricks.

* :ref:`commandline`

  Provides information on using the command line parsing library.

* :ref:`coding_standards`

  Details the LLVM coding standards and provides useful information on writing
  efficient C++ code.

* :doc:`HowToSetUpLLVMStyleRTTI`

  How to make ``isa<>``, ``dyn_cast<>``, etc. available for clients of your
  class hierarchy.

* :ref:`extending_llvm`

  Look here to see how to add instructions and intrinsics to LLVM.

* `Doxygen generated documentation <http://llvm.org/doxygen/>`_

  (`classes <http://llvm.org/doxygen/inherits.html>`_)
  (`tarball <http://llvm.org/doxygen/doxygen.tar.gz>`_)

* `ViewVC Repository Browser <http://llvm.org/viewvc/>`_

* :ref:`compiler_writer_info`

  A list of helpful links for compiler writers.
