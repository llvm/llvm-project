========================
LLVM 5.0.0 Release Notes
========================

.. contents::
    :local:


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 5.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* LLVM's ``WeakVH`` has been renamed to ``WeakTrackingVH`` and a new ``WeakVH``
  has been introduced.  The new ``WeakVH`` nulls itself out on deletion, but
  does not track values across RAUW.
  
* A new library named ``BinaryFormat`` has been created which holds a collection
  of code which previously lived in ``Support``.  This includes the
  ``file_magic`` structure and ``identify_magic`` functions, as well as all the
  structure and type definitions for DWARF, ELF, COFF, WASM, and MachO file
  formats.
  
* The tool ``llvm-pdbdump`` has been renamed ``llvm-pdbutil`` to better reflect
  its nature as a general purpose PDB manipulation / diagnostics tool that does
  more than just dumping contents.
  
* The ``BBVectorize`` pass has been removed. It was fully replaced and no
  longer used back in 2014 but we didn't get around to removing it. Now it is
  gone. The SLP vectorizer is the suggested non-loop vectorization pass.

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Changes to the LLVM IR
----------------------

* The datalayout string may now indicate an address space to use for
  the pointer type of alloca rather than the default of 0.

* Added speculatable attribute indicating a function which does has no
  side-effects which could inhibit hoisting of calls.

Changes to the Arm Targets
--------------------------

During this release the AArch64 target has:

* A much improved Global ISel at O0.
* Support for ARMv8.1 8.2 and 8.3 instructions.
* New scheduler information for ThunderX2.
* Some SVE type changes but not much more than that.
* Made instruction fusion more aggressive, resulting in speedups
  for code making use of AArch64 AES instructions. AES fusion has been
  enabled for most Cortex-A cores and the AArch64MacroFusion pass was moved
  to the generic MacroFusion pass.
* Added preferred function alignments for most Cortex-A cores.
* OpenMP "offload-to-self" base support.

During this release the ARM target has:

* Improved, but still mostly broken, Global ISel.
* Scheduling models update, new schedule for Cortex-A57.
* Hardware breakpoint support in LLDB.
* New assembler error handling, with spelling corrections and multiple
  suggestions on how to fix problems.
* Improved mixed ARM/Thumb code generation. Some cases in which wrong
  relocations were emitted have been fixed.
* Added initial support for mixed ARM/Thumb link-time optimization, using the
  thumb-mode target feature.

Changes to the MIPS Target
--------------------------

 During this release ...


Changes to the PowerPC Target
-----------------------------

* Additional support and exploitation of POWER ISA 3.0: vabsdub, vabsduh,
  vabsduw, modsw, moduw, modsd, modud, lxv, stxv, vextublx, vextubrx, vextuhlx,
  vextuhrx, vextuwlx, vextuwrx, vextsb2w, vextsb2d, vextsh2w, vextsh2d, and
  vextsw2d
  
* Implemented Optimal Code Sequences from The PowerPC Compiler Writer's Guide.

* Enable -fomit-frame-pointer by default.
  
* Improved handling of bit reverse intrinsic.
  
* Improved handling of memcpy and memcmp functions.
  
* Improved handling of branches with static branch hints.
  
* Improved codegen for atomic load_acquire.
  
* Improved block placement during code layout

* Many improvements to instruction selection and code generation




Changes to the X86 Target
-------------------------

* Added initial AMD Ryzen (znver1) scheduler support.

* Added support for Intel Goldmont CPUs.

* Add support for avx512vpopcntdq instructions.

* Added heuristics to convert CMOV into branches when it may be profitable.

* More aggressive inlining of memcmp calls.

* Improve vXi64 shuffles on 32-bit targets.

* Improved use of PMOVMSKB for any_of/all_of comparision reductions.

* Improved Silvermont, Sandybridge, and Jaguar (btver2) schedulers.

* Improved support for AVX512 vector rotations.

* Added support for AMD Lightweight Profiling (LWP) instructions.

* Avoid using slow LEA instructions.

* Use alternative sequences for multiply by constant.

* Improved lowering of strided shuffles.

* Improved the AVX512 cost model used by the vectorizer.

* Fix scalar code performance when AVX512 is enabled by making i1's illegal.

* Fixed many inline assembly bugs.

Changes to the AMDGPU Target
-----------------------------

* Initial gfx9 support

Changes to the AVR Target
-----------------------------

This release consists mainly of bugfixes and implementations of features
required for compiling basic Rust programs.

* Enable the branch relaxation pass so that we don't crash on large
  stack load/stores

* Add support for lowering bit-rotations to the native `ror` and `rol`
  instructions

* Fix bug where function pointers were treated as pointers to RAM and not
  pointers to program memory

* Fix broken code generaton for shift-by-variable expressions

* Support zero-sized types in argument lists; this is impossible in C,
  but possible in Rust

Changes to the OCaml bindings
-----------------------------

 During this release ...


Changes to the C API
--------------------

* Deprecated the ``LLVMAddBBVectorizePass`` interface since the ``BBVectorize``
  pass has been removed. It is now a no-op and will be removed in the next
  release. Use ``LLVMAddSLPVectorizePass`` instead to get the supported SLP
  vectorizer.


External Open Source Projects Using LLVM 5
==========================================

Zig Programming Language
------------------------

`Zig <http://ziglang.org>`_  is an open-source programming language designed
for robustness, optimality, and clarity. It integrates closely with C and is
intended to eventually take the place of C. It uses LLVM to produce highly
optimized native code and to cross-compile for any target out of the box. Zig
is in alpha; with a beta release expected in September.

LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X, FreeBSD and Windows and also Linux on ARM
and PowerPC (32/64 bit). Ports to other architectures like AArch64 and MIPS64
are underway.


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<http://llvm.org/>`_, in particular in the `documentation
<http://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <http://llvm.org/docs/#maillist>`_.
