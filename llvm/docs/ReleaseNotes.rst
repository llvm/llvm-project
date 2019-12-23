=========================
LLVM 10.0.0 Release Notes
=========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 10 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 10.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* The ISD::FP_ROUND_INREG opcode and related code was removed from SelectionDAG.
* Enabled MemorySSA as a loop dependency. Since
  `r370957 <https://reviews.llvm.org/rL370957>`_
  (`D58311 <https://reviews.llvm.org/D58311>`_ ``[MemorySSA & LoopPassManager]
  Enable MemorySSA as loop dependency. Update tests.``), the MemorySSA analysis
  is being preserved and used by a series of loop passes. The most significant
  use is in LICM, where the instruction hoisting and sinking relies on aliasing
  information provided by MemorySSA vs previously creating an AliasSetTracker.
  The LICM step of promoting variables to scalars still relies on the creation
  of an AliasSetTracker, but its use is reduced to only be enabled for loops
  with a small number of overall memory instructions. This choice was motivated
  by experimental results showing compile and run time benefits or replacing the
  AliasSetTracker usage with MemorySSA without any performance penalties.
  The fact that MemorySSA is now preserved by and available in a series of loop
  passes, also opens up opportunities for its use in those respective passes.
* The BasicBlockPass, BBPassManager and all their uses were deleted in
  `this revision <https://reviews.llvm.org/rG9f0ff0b2634bab6a5be8dace005c9eb24d386dd1>`_.

* The LLVM_BUILD_LLVM_DYLIB and LLVM_LINK_LLVM_DYLIB CMake options are no longer
  available on Windows.

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

* As per :ref:`LLVM Language Reference Manual <i_getelementptr>`,
  ``getelementptr inbounds`` can not change the null status of a pointer,
  meaning it can not produce non-null pointer given null base pointer, and
  likewise given non-null base pointer it can not produce null pointer; if it
  does, the result is a :ref:`poison value <poisonvalues>`.
  Since `r369789 <https://reviews.llvm.org/rL369789>`_
  (`D66608 <https://reviews.llvm.org/D66608>`_ ``[InstCombine] icmp eq/ne (gep
  inbounds P, Idx..), null -> icmp eq/ne P, null``) LLVM uses that for
  transformations. If the original source violates these requirements this
  may result in code being miscompiled. If you are using Clang front-end,
  Undefined Behaviour Sanitizer ``-fsanitize=pointer-overflow`` check
  will now catch such cases.


* Windows Control Flow Guard: the ``-cfguard`` option now emits CFG checks on
  indirect function calls. The previous behavior is still available with the 
  ``-cfguard-nochecks`` option. Note that this feature should always be used 
  with optimizations enabled.

* ``Callbacks`` have been added to ``CommandLine Options``.  These can
  be used to validate of selectively enable other options.

Changes to the LLVM IR
----------------------

* Unnamed function arguments now get printed with their automatically
  generated name (e.g. "i32 %0") in definitions. This may require front-ends
  to update their tests; if so there is a script utils/add_argument_names.py
  that correctly converted 80-90% of Clang tests. Some manual work will almost
  certainly still be needed.


Changes to building LLVM
------------------------

Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

 During this release ...


Changes to the PowerPC Target
-----------------------------

 During this release ...

Changes to the X86 Target
-------------------------

 During this release ...

* Less than 128 bit vector types, v2i32, v4i16, v2i16, v8i8, v4i8, and v2i8, are
  now stored in the lower bits of an xmm register and the upper bits are
  undefined. Previously the elements were spread apart with undefined bits in
  between them.
* v32i8 and v64i8 vectors with AVX512F enabled, but AVX512BW disabled will now
  be passed in ZMM registers for calls and returns. Previously they were passed
  in two YMM registers. Old behavior can be enabled by passing
  -x86-enable-old-knl-abi
* -mprefer-vector-width=256 is now the default behavior skylake-avx512 and later
  Intel CPUs. This tries to limit the use of 512-bit registers which can cause a
  decrease in CPU frequency on these CPUs. This can be re-enabled by passing
  -mprefer-vector-width=512 to clang or passing -mattr=-prefer-256-bit to llc.
* Deprecated the mpx feature flag for the Intel MPX instructions. There were no
  intrinsics for this feature. This change only this effects the results
  returned by getHostCPUFeatures on CPUs that implement the MPX instructions.
* The feature flag fast-partial-ymm-or-zmm-write which previously disabled
  vzeroupper insertion has been removed. It has been replaced with a vzeroupper
  feature flag which has the opposite polarity. So -vzeroupper has the same
  effect as +fast-partial-ymm-or-zmm-write.

Changes to the AMDGPU Target
-----------------------------

Changes to the AVR Target
-----------------------------

 During this release ...

Changes to the WebAssembly Target
---------------------------------

 During this release ...


Changes to the OCaml bindings
-----------------------------



Changes to the C API
--------------------
* C DebugInfo API ``LLVMDIBuilderCreateTypedef`` is updated to include an extra
  argument ``AlignInBits``, to facilitate / propagate specified Alignment information
  present in a ``typedef`` to Debug information in LLVM IR.


Changes to the Go bindings
--------------------------
* Go DebugInfo API ``CreateTypedef`` is updated to include an extra argument ``AlignInBits``,
  to facilitate / propagate specified Alignment information present in a ``typedef``
  to Debug information in LLVM IR.


Changes to the DAG infrastructure
---------------------------------

Changes to LLDB
===============

External Open Source Projects Using LLVM 10
===========================================

* A project...


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
