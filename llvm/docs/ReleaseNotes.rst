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

* The function attributes ``no-frame-pointer-elim`` and
  ``no-frame-pointer-elim-non-leaf`` have been replaced by ``frame-pointer``,
  which has 3 values: ``none``, ``non-leaf``, and ``all``. The values mean what
  functions should retain frame pointers.

* The inter-procedural analysis and optimization capabilities in the Attributor
  framework and pass have been substantially advanced (initial commit
  `D59918 <https://reviews.llvm.org/D59918>`_, `LLVM-Dev talk <https://youtu.be/CzWkc_JcfS0>`_).
  In this release, 19 different attributes are inferred, including 12 LLVM IR
  attributes and 7 "abstract" attributes, such as liveness. The Attributor is
  still under heavy development and disabled by default, to enable an early run
  pass ``-mllvm -attributor-disable=false`` to an invocation of clang.

* New matrix math intrinsics have been added to LLVM
  (see :ref:`LLVM Language Reference Manual <i_matrixintrinsics>`), together
  with the LowerMatrixIntrinsics pass. The pass lowers matrix intrinsics
  to a set of efficient vector instructions. The lowering pass is off
  by default and can be enabled by passing ``-mllvm -enable-matrix`` to an
  invocation of clang.


Changes to the LLVM IR
----------------------

* Unnamed function arguments now get printed with their automatically
  generated name (e.g. "i32 %0") in definitions. This may require front-ends
  to update their tests; if so there is a script utils/add_argument_names.py
  that correctly converted 80-90% of Clang tests. Some manual work will almost
  certainly still be needed.

* A new `freeze` instruction is added. The `freeze` instruction is used to stop
  IR-level propagation of undef and poison values. Currently its support is
  preliminary; a freeze-equivalent operation for SelDag/MIR needs to be added.


Changes to building LLVM
------------------------

...

Changes to the AArch64 Backend
------------------------------

- Added support for Cortex-A65, Cortex-A65AE, Neoverse E1 and Neoverse N1 cores.
- With a few more bugs fixed in the LLVM 10 release, clang-cl can now target windows-on-Arm well, demonstrated by building complex pieces of software such as Chromium and the Electron framework.
- Support for -fpatchable-function-entry was added.

Changes to the ARM Backend
--------------------------

- Optimized Armv8.1-M code generation, including generating Low Overhead Loops.
- Added auto-vectorization for the Armv8.1-M MVE vector extension.
- Support was added for inline asm constraints s,j,x,N,O.


Changes to the MIPS Target
--------------------------

* Improved support for ``octeon`` and added support for ``octeon+``
  MIPS-family CPU.
* ``min``, ``max``, ``umin``, ``umax`` atomics now supported on MIPS targets.
* Now PC-relative relocations are generated for ``.eh_frame`` sections when
  possible. That allows to link MIPS binaries without having to pass the
  ``-Wl,-z,notext`` option.
* Fix evaluating J-format branch (``j``, ``jal``, ...) targets when the
  instruction is not in the first 256 MB region.
* Fixed ``jal``, ``sc``, ``scs``, ``ll``, ``lld``, ``la``, ``lw``, ``sw``
  instructions expanding. Now they accept more types of expression as arguments,
  correctly handle load/store for ``XGOT`` model, expand using less instructions
  or registers.
* Initial MIPS support has been added to ``llvm-exegesis``.
* Generates ``_mcount`` calls using proper MIPS ABI.
* Improved support of GlobalISel instruction selection framework. This feature
  is still in experimental state for MIPS targets though.

Changes to the PowerPC Target
-----------------------------

Optimization:

  *  Improved register pressure estimates in the loop vectorizer based on type
  *  Improved the PowerPC cost model for the vectorizer
  *  Enabled vectorization of math routines on PowerPC using MASSV (Mathematical Acceleration SubSystem) library

compiler-rt:

  *  Added/improved conversion functions from IBM long double to 128-bit integers

Codegen:

   *  Optimized memory access instructions in loops (pertaining to update-form instructions and address computation)
   *  Added options to disable hoisting instructions to hotter blocks based on statically or profile-based block hotness estimates
   *  Code generation improvements (particularly with floating point and vector code as well as handling condition registers)
   *  Various infrastructural improvements, code refactoring, and bug fixes
   *  Optimized handling of control flow based on multiple comparison of same values

Tools:

  *   llvm-readobj supports displaying file header, section headers, symbol table and relocation entries for XCOFF object files
  *   llvm-objdump supports disassembling physical sections for XCOFF object files


Changes to the SystemZ Target
-----------------------------

* Added support for the ``-march=z15`` and ``-mtune=z15`` command line options
  (as aliases to the existing ``-march=arch13`` and ``-mtune=arch13`` options).
* Added support for the ``-march=native`` command line option.
* Added support for the ``-mfentry``, ``-mnop-mcount``, and ``-mrecord-mcount``
  command line options.
* Added support for the GHC calling convention.
* Miscellaneous codegen enhancements, in particular to enable better
  reuse of condition code values and improved use of conditional
  move instructions.

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

* __attribute__((used)) no longer implies that a symbol is exported, for
  consistency with other targets.
* Multivalue function signatures are now supported in WebAssembly object files
* The new `atomic.fence` instruction is now supported
* Thread-Local Storage (TLS) is now supported.
* SIMD support is significantly expanded.

Changes to the Windows Target
-----------------------------

* Fixed section relative relocations in .debug_frame in DWARF debug info

Changes to the RISC-V Target
----------------------------

New Features:
* The Machine Outliner is now supported, but not enabled by default.
* Shrink-wrapping is now supported.
* The Machine Scheduler has been enabled and scheduler descriptions for the
  Rocket micro-architecture have been added, covering both 32- and 64-bit Rocket
  cores.
* This release lays the groundwork for enabling LTO in a future LLVM release.
  In particular, LLVM now uses a new `target-abi` module metadata item to
  represent the chosen RISC-V psABI variant. Frontends should add this module
  flag to prevent ABI lowering problems when LTO is enabled in a future LLVM
  release.
* Support has been added for assembling RVC HINT instructions.
* Added code lowering for half-precision floats.
* The `fscsr` and `frcsr` (`fssr`, `frsr`) obsolete aliases have been added to
  the assembler for use in legacy code.
* The stack can now be realigned even when there are variable-sized objects in
  the same frame.
* fastcc is now supported. This is a more efficient, unstandardised, calling
  convention for calls to private leaf functions in the same IR module.
* llvm-objdump now supports `-M no-aliases` and `-M numeric` for altering the
  dumped assembly. These match the behaviour of GNU objdump, respectively
  disabling instruction aliases and printing the numeric register names rather
  than the ABI register names.

Improvements:
* Trap and Debugtrap now lower to RISC-V-specific trap instructions.
* LLVM IR Inline assembly now supports using ABI register names and using
  floating point registers in constraints.
* Stack Pointer adjustments have been changed to better match RISC-V's immediates.
* `ra` (`x1`) can now be used as a callee-saved register.
* The assembler now suggests spelling corrections for unknown assembly
  mnemonics.
* Stack offsets of greater than 32-bits are now accepted on RV64.
* Variadic functions can now be tail-call optimised, as long as they do not use
  stack memory for passing arguments.
* Code generation has been changed for 32-bit arithmetic operations on RV64 to
  reduce sign-extensions.

Bug Fixes:
* There was an issue with register preservation after calls in interrupt
  handlers, where some registers were marked as preserved even though they were
  not being preserved by the call. This has been corrected, and now only
  callee-saved registers are live over a function call in an interrupt handler
  (just like calls in regular functions).
* Atomic instructions now only accept GPRs (plus an offset) in memory operands.
* Fixed some issues with evaluation of relocations and fixups.
* The error messages around missing RISC-V extensions in the assembler have been
  improved.
* The error messages around unsupported relocations have been improved.
* Non-PIC code no longer forces Local Exec TLS.
* There have been some small changes to the code generation for atomic
  operations.
* RISC-V no longer emits incorrect CFI directives in function prologs and
  epilogs.
* RV64 no longer clears the upper bits when returning complex types from
  libcalls using the LP64 psABI.

Compiler-RT:
* RISC-V (both 64-bit and 32-bit) is now supported by compiler-rt, allowing
  crtbegin and crtend to be built.
* The Sanitizers now support 64-bit RISC-V on linux.

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

* Improved support for building with MinGW

* Initial support for debugging Windows ARM and ARM64 binaries

* Improved error messages in the expression evaluator.

* Tab completions for command options now also provide a description for each option.

* Fixed that printing structs/classes with the `expression` command sometimes did not
  print the members/contents of the class.

* Improved support for using classes with bit-field members in the expression evaluator.

* Greatly improved support for DWARF v5.

External Open Source Projects Using LLVM 10
===========================================

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
