============================
LLVM |release| Release Notes
============================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release |release|.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `Discourse forums
<https://discourse.llvm.org>`_ is a good place to ask
them.

Note that if you are reading this file from a Git checkout or the main
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

* ...

Update on required toolchains to build LLVM
-------------------------------------------

With LLVM 17.x we raised the version requirement of CMake used to build LLVM.
The new requirements are as follows:

* CMake >= 3.20.0

Changes to the LLVM IR
----------------------

* Typed pointers are no longer supported and the ``-opaque-pointers`` option
  has been removed. See the `opaque pointers <OpaquePointers.html>`__
  documentation for migration instructions.

* The ``nofpclass`` attribute was introduced. This allows more
  optimizations around special floating point value comparisons.

* Introduced new ``llvm.ldexp`` and ``llvm.experimental.constrained.ldexp`` intrinsics.

* Introduced new ``llvm.frexp`` intrinsic.

* The constant expression variants of the following instructions have been
  removed:

  * ``select``

* Introduced a set of experimental `convergence control intrinsics
  <ConvergentOperations.html>`__ to explicitly define the semantics of convergent
  operations.

Changes to LLVM infrastructure
------------------------------

* The legacy optimization pipeline (``PassManagerBuilder.h``) has been removed.
  See the `new pass manager docs <https://llvm.org/docs/NewPassManager.html>`_
  for how to use the new pass manager APIs.

* Alloca merging in the inliner has been removed, since it only worked with the
  legacy inliner pass. Backend stack coloring should handle cases alloca
  merging initially set out to handle.

* InstructionSimplify APIs now require instructions be inserted into a
  parent function.

* A new FatLTO pipeline was added to support generating object files that have
  both machine code and LTO compatible bitcode. See the :doc:`FatLTO`
  documentation and the original
  `RFC  <https://discourse.llvm.org/t/rfc-ffat-lto-objects-support/63977>`_
  for more details.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

* Named arguments are supported. Arguments can be specified in the form of
  ``name=value``.

Changes to Optimizations
----------------------------------------

* :ref:`llvm.assume <int_assume>` now recognizes certain
  floating-point tests. e.g. ``__builtin_assume(!isnan(x))``

Changes to the AArch64 Backend
------------------------------

* Added Assembly Support for the 2022 A-profile extensions FEAT_GCS (Guarded
  Control Stacks), FEAT_CHK (Check Feature Status), and FEAT_ATS1A.
* Support for preserve_all calling convention is added.
* Added support for missing arch extensions in the assembly directives
  ``.arch <level>+<ext>`` and ``.arch_extension``.
* Fixed handling of ``.arch <level>`` in assembly, without using any ``+<ext>``
  suffix. Previously this had no effect at all if no extensions were supplied.
  Now ``.arch <level>`` can be used to enable all the extensions that are
  included in a higher level than what is specified on the command line,
  or for disabling unwanted extensions if setting it to a lower level.
  This fixes `PR32873 <https://github.com/llvm/llvm-project/issues/32220>`.

Changes to the AMDGPU Backend
-----------------------------
* More fine-grained synchronization around barriers for newer architectures
  (gfx90a+, gfx10+). The AMDGPU backend now omits previously automatically
  generated waitcnt instructions before barriers, allowing for more precise
  control. Users must now use memory fences to implement fine-grained
  synchronization strategies around barriers. Refer to `AMDGPU memory model
  <AMDGPUUsage.html#memory-model>`__.

* Address space 7, used for *buffer fat pointers* has been added.
  It is non-integral and has 160-bit pointers (a 128-bit raw buffer resource and a
  32-bit offset) and 32-bit indices. This is part of ongoing work to improve
  the usability of buffer operations. Refer to `AMDGPU address spaces
  <AMDGPUUsage.html#address-spaces>`__.

* Address space 8, used for *buffer resources* has been added.
  It is non-integral and has 128-bit pointers, which correspond to buffer
  resources in the underlying hardware. These pointers should not be used with
  `getelementptr` or other LLVM memory instructions, and can be created with
  the `llvm.amdgcn.make.buffer.rsrc` intrinsic. Refer to `AMDGPU address spaces
  <AMDGPUUsage.html#address_spaces>`__.

* New versions of the intrinsics for working with buffer resources have been added.
  These `llvm.amdgcn.*.ptr.[t]buffer.*` intrinsics have the same semantics as
  the old `llvm.amdgcn.*.[t]buffer.*` intrinsics, except that their `rsrc`
  arguments are represented by a `ptr addrspace(8)` instead of a `<4 x i32>`. This
  improves the interaction between AMDGPU buffer operations and the LLVM memory
  model, and so the non `.ptr` intrinsics are deprecated.

* Backend now performs range merging of "amdgpu-waves-per-eu" attribute based on
  known callers.

* Certain :ref:`atomicrmw <i_atomicrmw>` operations are now optimized by
  performing a wave reduction if the access is uniform by default.

* Removed ``llvm.amdgcn.atomic.inc`` and ``llvm.amdgcn.atomic.dec``
  intrinsics. :ref:`atomicrmw <i_atomicrmw>` should be used instead
  with ``uinc_wrap`` and ``udec_wrap``.

* Added llvm.amdgcn.log.f32 intrinsic. This provides direct access to
  v_log_f32.

* Added llvm.amdgcn.exp2.f32 intrinsic. This provides direct access to
  v_exp_f32.

* llvm.log2.f32, llvm.log10.f32, and llvm.log.f32 are now lowered
  accurately. Use llvm.amdgcn.log.f32 to access the old behavior for
  llvm.log2.f32.

* llvm.exp2.f32 and llvm.exp.f32 are now lowered accurately. Use
  llvm.amdgcn.exp2.f32 to access the old behavior for llvm.exp2.f32.

* Implemented new 1ulp IEEE lowering strategy for float reciprocal
  which saves 2 instructions. This is used by default for OpenCL on
  gfx9+. With ``contract`` flags, this will fold into a 1 ulp rsqrt.

* Implemented new 2ulp IEEE lowering strategy for float
  reciprocal. This is used by default for OpenCL on gfx9+.

* `llvm.sqrt.f64` is now lowered correctly. Use `llvm.amdgcn.sqrt.f64`
  for raw instruction access.

* Deprecate `llvm.amdgcn.ldexp` intrinsic. :ref:`llvm.ldexp <int_ldexp>`
  should be used instead.

Changes to the ARM Backend
--------------------------

- The hard-float ABI is now available in Armv8.1-M configurations that
  have integer MVE instructions (and therefore have FP registers) but
  no scalar or vector floating point computation.

- The ``.arm`` directive now aligns code to the next 4-byte boundary, and
  the ``.thumb`` directive aligns code to the next 2-byte boundary.

Changes to the AVR Backend
--------------------------

* ...

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

* ...

Changes to the LoongArch Backend
--------------------------------

* Adds assembler/disassembler support for the LSX, LASX, LVZ and LBT ISA extensions.
* The ``lp64s`` ABI is supported now and has been tested on Rust bare-matal target.
* A target feature ``ual`` is introduced to allow unaligned memory accesses and
  this feature is enabled by default for generic 64-bit processors.
* Adds support for the ``large`` code model, which is equivalent to GCC's ``extreme`` one.
* Assorted codegen improvements.
* llvm-objcopy now supports processing LoongArch objects.

Changes to the MIPS Backend
---------------------------

* ...

Changes to the PowerPC Backend
------------------------------

* Improved code sequence of materializing 64-bit immediate numbers, expanding
  ``is.fpclass`` intrinsic and forwarding stores.
* Implemented DFP instructions (for use via inline asm).
* Improved code gen for vector add.
* Added ability to show statistics of number of entries in the TOC.
* Added Binary Coded Decimal Assist instructions (for use via inline asm).
* Added basic support for vector functions in GlobalISel.
* Added additional X-Form load and store instruction generation for TLS accesses.
* PPC64LE backend is added to JITLink.
* Added various bug fixes and optimizations.
* Added function pointer alignment to the DataLayout for Power, which lets us
  make more informed choices about what this alignment defaults to for various 
  purposes (e.g., C++ pointers to member). If the target ABI uses function
  descriptor objects, this is the alignment we would emit the descriptor with.
  Otherwise, a function pointer points to a global entry point, so this is at
  least the alignment for code on Power (i.e., 4-bytes).

AIX Support/improvements:


* A new option ``-mxcoff-roptr`` is added to ``clang`` and ``llc``. When this
  option is present, constant objects with relocatable address values are put
  into the RO data section. This option should be used with the
  ``-fdata-sections`` option, and is not supported with ``-fno-data-sections``.

* Taught the profile runtime to check for a build-id string. Build-id strings
  can be created via the ``-mxcoff-build-id`` option.

* Removed ``-ppc-quadword-atomics`` which only affected lock-free quadword
  atomics on AIX. Now backend generates lock-free quadword atomics code on AIX
  by default. To support lock-free quadword atomics in libatomic, the OS level
  must be at least AIX 7.2 TL5 SP3 with libc++.rte of version 17.1.1 or above
  installed.

* Integrated assembler is enabled by default on AIX.
* System assembler is always used to compile assembly files on AIX.
* Added support for local-exec TLS.
* Added a new option, ``--traceback-table``, to ``llvm-objdump`` to print out
  the traceback table information for XCOFF object files.
* Added ``llvm-ar`` object mode options ``-X32``, ``-X64``, ``-X32-64``,
  and ``-Xany``.
* Changed the default name of the text-section csect to be an empty string
  instead of ``.text``. This change does not affect the behaviour
  of the program.
* Fixed a problem when the personality routine for the legacy AIX ``xlclang++``
  compiler uses the stack slot to pass the exception object to the landing pad.
  Runtime routine ``__xlc_exception_handle()`` invoked by the landing pad to
  retrieve the exception object now skips frames not associated with functions
  that are C++ EH-aware because the compiler sometimes generates a wrapper of
  ``__xlc_exception_handle()`` for optimization purposes.

Changes to the RISC-V Backend
-----------------------------

* Assembler support for version 1.0.1 of the Zcb extension was added.
* Zca, Zcf, and Zcd extensions were upgraded to version 1.0.1.
* vsetvli intrinsics no longer have side effects. They may now be combined,
  moved, deleted, etc. by optimizations.
* Adds support for the vendor-defined XTHeadBa (address-generation) extension.
* Adds support for the vendor-defined XTHeadBb (basic bit-manipulation) extension.
* Adds support for the vendor-defined XTHeadBs (single-bit) extension.
* Adds support for the vendor-defined XTHeadCondMov (conditional move) extension.
* Adds support for the vendor-defined XTHeadMac (multiply-accumulate instructions) extension.
* Added support for the vendor-defined XTHeadMemPair (two-GPR memory operations)
  extension disassembler/assembler.
* Added support for the vendor-defined XTHeadMemIdx (indexed memory operations)
  extension disassembler/assembler.
* Added support for the vendor-defined Xsfvcp (SiFive VCIX) extension
  disassembler/assembler.
* Added support for the vendor-defined Xsfcie (SiFive CIE) extension
  disassembler/assembler.
* Support for the now-ratified Zawrs extension is no longer experimental.
* Adds support for the vendor-defined XTHeadCmo (cache management operations) extension.
* Adds support for the vendor-defined XTHeadSync (multi-core synchronization instructions) extension.
* Added support for the vendor-defined XTHeadFMemIdx (indexed memory operations for floating point) extension.
* Assembler support for RV64E was added.
* Assembler support was added for the experimental Zicond (integer conditional
  operations) extension.
* I, F, D, and A extension versions have been update to the 20191214 spec versions.
  New version I2.1, F2.2, D2.2, A2.1. This should not impact code generation.
  Immpacts versions accepted in ``-march`` and reported in ELF attributes.
* Changed the ShadowCallStack register from ``x18`` (``s2``) to ``x3``
  (``gp``). Note this breaks the existing non-standard ABI for ShadowCallStack
  on RISC-V, but conforms with the new "platform register" defined in the
  RISC-V psABI (for more details see the
  `psABI discussion <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/issues/370>`_).
* Added support for Zfa extension version 0.2.
* Updated support experimental vector crypto extensions to version 0.5.1 of
  the specification.
* Removed N extension (User-Level Interrupts) CSR names in the assembler.
* ``RISCV::parseCPUKind`` and ``RISCV::checkCPUKind`` were merged into a single
  ``RISCV::parseCPU``. The ``CPUKind`` enum is no longer part of the
  RISCVTargetParser.h interface. Similar for ``parseTuneCPUkind`` and
  ``checkTuneCPUKind``.
* Add sifive-x280 processor.
* Zve32f is no longer allowed with Zfinx. Zve64d is no longer allowed with
  Zdinx.
* Assembly support was added for the experimental Zfbfmin (scalar BF16
  conversions), Zvfbfmin (vector BF16 conversions), and Zvfbfwma (vector BF16
  widening mul-add) extensions.
* Added assembler/disassembler support for the experimental Zacas (atomic
  compare-and-swap) extension.
* Zvfh extension version was upgraded to 1.0 and is no longer experimental.

Changes to the WebAssembly Backend
----------------------------------

* Function annotations (``__attribute__((annotate(<name>)))``)
  now generate custom sections in the Wasm output file. A custom section
  for each unique name will be created that contains each function
  index the annotation applies to.

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

* ``__builtin_unpredictable`` (unpredictable metadata in LLVM IR), is handled by X86 Backend.
  ``X86CmovConversion`` pass now respects this builtin and does not convert CMOVs to branches.
* Add support for the ``PBNDKB`` instruction.
* Support ISA of ``SHA512``.
* Support ISA of ``SM3``.
* Support ISA of ``SM4``.
* Support ISA of ``AVX-VNNI-INT16``.
* ``-mcpu=graniterapids-d`` is now supported.

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

* The python bindings have been removed.


Changes to the C API
--------------------

* ``LLVMContextSetOpaquePointers``, a temporary API to pin to legacy typed
  pointer, has been removed.
* Functions for adding legacy passes like ``LLVMAddInstructionCombiningPass``
  have been removed.
* Removed ``LLVMPassManagerBuilderRef`` and functions interacting with it.
  These belonged to the no longer supported legacy pass manager.
* Functions for initializing legacy passes like ``LLVMInitializeInstCombine``
  have been removed. Calls to such functions can simply be dropped, as they are
  no longer necessary.
* ``LLVMPassRegistryRef`` and ``LLVMGetGlobalPassRegistry``, which were only
  useful in conjunction with initialization functions, have been removed.
* As part of the opaque pointer transition, ``LLVMGetElementType`` no longer
  gives the pointee type of a pointer type.
* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstSelect``

Changes to the CodeGen infrastructure
-------------------------------------

* ``llvm.memcpy``, ``llvm.memmove`` and ``llvm.memset`` are now
  expanded into loops by default for targets which do not report the
  corresponding library function is available.

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

* The DWARFv5 feature of attaching ``DW_AT_default_value`` to defaulted template
  parameters will now be available in any non-strict DWARF mode and in a wider
  range of cases than previously.
  (`D139953 <https://reviews.llvm.org/D139953>`_,
  `D139988 <https://reviews.llvm.org/D139988>`_)

* The ``DW_AT_name`` on ``DW_AT_typedef``\ s for alias templates will now omit
  defaulted template parameters. (`D142268 <https://reviews.llvm.org/D142268>`_)

* The experimental ``@llvm.dbg.addr`` intrinsic has been removed (`D144801
  <https://reviews.llvm.org/D144801>`_). IR inputs with this intrinsic are
  auto-upgraded to ``@llvm.dbg.value`` with ``DW_OP_deref`` appended to the
  ``DIExpression`` (`D144793 <https://reviews.llvm.org/D144793>`_).

* When a template class annotated with the ``[[clang::preferred_name]]`` attribute
  were to appear in a ``DW_AT_type``, the type will now be that of the preferred_name
  instead. This change is only enabled when compiling with `-glldb`.
  (`D145803 <https://reviews.llvm.org/D145803>`_)

Changes to the LLVM tools
---------------------------------
* llvm-lib now supports the /def option for generating a Windows import library from a definition file.

* Made significant changes to JSON output format of `llvm-readobj`/`llvm-readelf`
  to improve correctness and clarity.

Changes to LLDB
---------------------------------

* In the results of commands such as ``expr`` and ``frame var``, type summaries will now
  omit defaulted template parameters. The full template parameter list can still be
  viewed with ``expr --raw-output``/``frame var --raw-output``. (`D141828 <https://reviews.llvm.org/D141828>`_)

* LLDB is now able to show the subtype of signals found in a core file. For example
  memory tagging specific segfaults such as ``SIGSEGV: sync tag check fault``.

* LLDB can now display register fields if they are described in target XML sent
  by a debug server such as ``gdbserver`` (``lldb-server`` does not currently produce
  this information). Fields are only printed when reading named registers, for
  example ``register read cpsr``. They are not shown when reading a register set,
  ``register read -s 0``.

* A new command ``register info`` was added. This command will tell you everything that
  LLDB knows about a register. Based on what LLDB already knows and what the debug
  server tells it. Including but not limited to, the size, where it is read from and
  the fields that the register contains.

* AArch64 Linux targets now provide access to the Thread Local Storage
  register ``tpidr``.

Changes to Sanitizers
---------------------
* For Darwin users that override weak symbols, note that the dynamic linker will
  only consider symbols in other mach-o modules which themselves contain at
  least one weak symbol. A consequence is that if your program or dylib contains
  an intended override of a weak symbol, then it must contain at least one weak
  symbol as well for the override to take effect.

  Example:

  .. code-block:: c

    // Add this to make sure your override takes effect
    __attribute__((weak,unused)) unsigned __enableOverrides;

    // Example override
    extern "C" const char *__asan_default_options() { ... }

Changes to BOLT
---------------
* Initial RISC-V (RV64GC) target support was added.
* DWARFRewriter got new mechanism for more flexible handling of debug
  information. It raises debug information to IR level before performing
  updates, and IR is written out to the binary after updates are applied.
* Stale profile matching was added under a flag `--infer-stale-profile`.
  It requires the use of a YAML profile, produced by perf2bolt using `-w`
  flag, or with `--profile-format=yaml`.


Other Changes
-------------

* ``llvm::demangle`` now takes a ``std::string_view`` rather than a
  ``const std::string&``. Be careful passing temporaries into
  ``llvm::demangle`` that don't outlive the expression using
  ``llvm::demangle``.

External Open Source Projects Using LLVM 15
===========================================

* A project...

Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Git version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `Discourse forums <https://discourse.llvm.org>`_.
