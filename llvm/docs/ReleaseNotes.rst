========================
LLVM 8.0.0 Release Notes
========================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 8.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://releases.llvm.org/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Minimum Required Compiler Version
=================================
As `discussed on the mailing list
<https://lists.llvm.org/pipermail/llvm-dev/2019-January/129452.html>`_,
building LLVM will soon require more recent toolchains as follows:

============= ====
Clang         3.5
Apple Clang   6.0
GCC           5.1
Visual Studio 2017
============= ====

A new CMake check when configuring LLVM provides a soft-error if your
toolchain will become unsupported soon. You can opt out of the soft-error by
setting the ``LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN`` CMake variable to
``ON``.


Known Issues
============

These are issues that couldn't be fixed before the release. See the bug reports
for the latest status.

* `PR40547 <https://llvm.org/pr40547>`_ Clang gets miscompiled by trunk GCC.

* `PR40761 <https://llvm.org/pr40761>`_ "asan-dynamic" doesn't work on FreeBSD.


Non-comprehensive list of changes in this release
=================================================

* The **llvm-cov** tool can now export lcov trace files using the
  `-format=lcov` option of the `export` command.

* The ``add_llvm_loadable_module`` CMake macro has been removed.  The
  ``add_llvm_library`` macro with the ``MODULE`` argument now provides the same
  functionality.  See `Writing an LLVM Pass
  <WritingAnLLVMPass.html#setting-up-the-build-environment>`_.

* For MinGW, references to data variables that might need to be imported
  from a dll are accessed via a stub, to allow the linker to convert it to
  a dllimport if needed.

* Added support for labels as offsets in ``.reloc`` directive.

* Support for precise identification of X86 instructions with memory operands,
  by using debug information. This supports profile-driven cache prefetching.
  It is enabled with the ``-x86-discriminate-memops`` LLVM Flag.

* Support for profile-driven software cache prefetching on X86. This is part of
  a larger system, consisting of: an offline cache prefetches recommender,
  AutoFDO tooling, and LLVM. In this system, a binary compiled with
  ``-x86-discriminate-memops`` is run under the observation of the recommender.
  The recommender identifies certain memory access instructions by their binary
  file address, and recommends a prefetch of a specific type (NTA, T0, etc) be
  performed at a specified fixed offset from such an instruction's memory
  operand. Next, this information needs to be converted to the AutoFDO syntax
  and the resulting profile may be passed back to the compiler with the LLVM
  flag ``-prefetch-hints-file``, together with the exact same set of
  compilation parameters used for the original binary. More information is
  available in the `RFC
  <https://lists.llvm.org/pipermail/llvm-dev/2018-November/127461.html>`_.

* Windows support for libFuzzer (x86_64).

Changes to the LLVM IR
----------------------

* Function attribute ``speculative_load_hardening`` has been introduced to
  allow indicating that `Speculative Load Hardening
  <SpeculativeLoadHardening.html>`_ must be enabled for the function body.


Changes to the JIT APIs
-----------------------

The ORC (On Request Compilation) JIT APIs have been updated to support
concurrent compilation. The existing (non-concurrent) ORC layer classes and
related APIs are deprecated, have been renamed with a "Legacy" prefix (e.g.
LegacyIRCompileLayer). The deprecated clasess will be removed in LLVM 9.

An example JIT stack using the concurrent ORC APIs, called LLJIT, has been
added (see include/llvm/ExecutionEngine/Orc/LLJIT.h). The lli tool has been
updated to use LLJIT.

MCJIT and ExecutionEngine continue to be supported, though ORC should be
preferred for new projects.

Changes to the C++ APIs
-----------------------

Three of the IR library methods related to debugging information for
functions and methods have changed their prototypes:

  DIBuilder::createMethod
  DIBuilder::createFunction
  DIBuilder::createTempFunctionFwdDecl

In all cases, several individual parameters were removed, and replaced
by a single 'SPFlags' (subprogram flags) parameter. The individual
parameters are: 'isLocalToUnit'; 'isDefinition'; 'isOptimized'; and
for 'createMethod', 'Virtuality'.  The new 'SPFlags' parameter has a
default value equivalent to passing 'false' for the three 'bool'
parameters, and zero (non-virtual) to the 'Virtuality' parameter.  For
any old-style API call that passed 'true' or a non-zero virtuality to
these methods, you will need to substitute the correct 'SPFlags' value.
The helper method 'DISubprogram::toSPFlags()' might be useful in making
this conversion.

Changes to the AArch64 Target
-----------------------------

* Support for Speculative Load Hardening has been added.

* Initial support for the Tiny code model, where code and its statically
  defined symbols must live within 1MB of each other.

* Added support for the ``.arch_extension`` assembler directive, just like
  on ARM.


Changes to the Hexagon Target
-----------------------------

* Added support for Hexagon/HVX V66 ISA.


Changes to the MIPS Target
--------------------------

* Improved support of GlobalISel instruction selection framework.

* Implemented emission of ``R_MIPS_JALR`` and ``R_MICROMIPS_JALR``
  relocations. These relocations provide hints to a linker for optimization
  of jumps to protected symbols.

* ORC JIT has been supported for MIPS and MIPS64 architectures.

* Assembler now suggests alternative MIPS instruction mnemonics when
  an invalid one is specified.

* Improved support for MIPS N32 ABI.

* Added new instructions (``pll.ps``, ``plu.ps``, ``cvt.s.pu``,
  ``cvt.s.pl``, ``cvt.ps``, ``sigrie``).

* Numerous bug fixes and code cleanups.


Changes to the PowerPC Target
-----------------------------

* Switched to non-PIC default

* Deprecated Darwin support

* Enabled Out-of-Order scheduling for P9

* Better overload rules for compatible vector type parameter

* Support constraint 'wi', modifier 'x' and VSX registers in inline asm

* More ``__float128`` support

* Added new builtins like vector int128 ``pack``/``unpack`` and
  ``stxvw4x.be``/``stxvd2x.be``

* Provided significant improvements to the automatic vectorizer

* Code-gen improvements (especially for Power9)

* Fixed some long-standing bugs in the back end

* Added experimental prologue/epilogue improvements

* Enabled builtins tests in compiler-rt

* Add ``___fixunstfti``/``floattitf`` in compiler-rt to support conversion
  between IBM double-double and unsigned int128

* Disable randomized address space when running the sanitizers on Linux ppc64le

* Completed support in LLD for ELFv2

* Enabled llvm-exegesis latency mode for PPC


Changes to the SystemZ Target
-----------------------------

* A number of bugs related to C/C++ language vector extension support were
  fixed: the ``-mzvector`` option now actually enables the ``__vector`` and
  ``__bool`` keywords, the ``vec_step`` intrinsic now works, and the
  ``vec_insert_and_zero`` and ``vec_orc`` intrinsics now generate correct code.

* The ``__float128`` keyword, which had been accidentally enabled in some
  earlier releases, is now no longer supported.  On SystemZ, the ``long double``
  data type itself already uses the IEEE 128-bit floating-point format.

* When the compiler inlines ``strcmp`` or ``memcmp``, the generated code no
  longer returns ``INT_MIN`` as the negative result value under any
  circumstances.

* Various code-gen improvements, in particular related to improved
  auto-vectorization, inlining, and instruction scheduling.


Changes to the X86 Target
-------------------------

* Machine model for AMD bdver2 (Piledriver) CPU was added. It is used to support
  instruction scheduling and other instruction cost heuristics.

* New AVX512F gather and scatter intrinsics were added that take a <X x i1> mask
  instead of a scalar integer. This removes the need for a bitcast in IR. The
  new intrinsics are named like the old intrinsics with ``llvm.avx512.``
  replaced with ``llvm.avx512.mask.``. The old intrinsics will be removed in a
  future release.

* Added ``cascadelake`` as a CPU name for -march. This is ``skylake-avx512``
  with the addition of the ``avx512vnni`` instruction set.

* ADCX instruction will no longer be emitted. This instruction is rarely better
  than the legacy ADC instruction and just increased code size.


Changes to the WebAssembly Target
---------------------------------

The WebAssembly target is no longer "experimental"! It's now built by default,
rather than needing to be enabled with LLVM_EXPERIMENTAL_TARGETS_TO_BUILD.

The object file format and core C ABI are now considered stable. That said,
the object file format has an ABI versioning capability, and one anticipated
use for it will be to add support for returning small structs as multiple
return values, once the underlying WebAssembly platform itself supports it.
Additionally, multithreading support is not yet included in the stable ABI.


Changes to the Nios2 Target
---------------------------

* The Nios2 target was removed from this release.


Changes to LLDB
===============

* Printed source code is now syntax highlighted in the terminal (only for C
  languages).

* The expression command now supports tab completing expressions.


External Open Source Projects Using LLVM 8
==========================================

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

Open Dylan Compiler
-------------------

`Dylan <https://opendylan.org/>`_ is a multi-paradigm functional
and object-oriented programming language.  It is dynamic while
providing a programming model designed to support efficient machine
code generation, including fine-grained control over dynamic and
static behavior. Dylan also features a powerful macro facility for
expressive metaprogramming.

The Open Dylan compiler can use LLVM as one of its code-generating
back-ends, including full support for debug info generation. (Open
Dylan generates LLVM bitcode directly using a native Dylan IR and
bitcode library.) Development of a Dylan debugger and interactive REPL
making use of the LLDB libraries is in progress.

Zig Programming Language
------------------------

`Zig <https://ziglang.org>`_  is a system programming language intended to be
an alternative to C. It provides high level features such as generics, compile
time function execution, and partial evaluation, while exposing low level LLVM
IR features such as aliases and intrinsics. Zig uses Clang to provide automatic
import of .h symbols, including inline functions and simple macros. Zig uses
LLD combined with lazily building compiler-rt to provide out-of-the-box
cross-compiling for all supported targets.


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
