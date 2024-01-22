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

Changes to the LLVM IR
----------------------

* The `llvm.stacksave` and `llvm.stackrestore` intrinsics now use
  an overloaded pointer type to support non-0 address spaces.
* The constant expression variants of the following instructions have been
  removed:

  * ``and``
  * ``or``
  * ``lshr``
  * ``ashr``
  * ``zext``
  * ``sext``
  * ``fptrunc``
  * ``fpext``
  * ``fptoui``
  * ``fptosi``
  * ``uitofp``
  * ``sitofp``

* Added `llvm.exp10` intrinsic.

* Added a ``code_model`` attribute for the `global variable <LangRef.html#global-variables>`_.

Changes to LLVM infrastructure
------------------------------

* Minimum Clang version to build LLVM in C++20 configuration has been updated to clang-17.0.6.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

* Added constructs for debugging TableGen files:

  * `dump` keyword to dump messages to standard error, see
     https://github.com/llvm/llvm-project/pull/68793.
  * `!repr` bang operator to inspect the content of values, see
     https://github.com/llvm/llvm-project/pull/68716.

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* Added support for Cortex-A520, Cortex-A720 and Cortex-X4 CPUs.

* Neoverse-N2 was incorrectly marked as an Armv8.5a core. This has been
  changed to an Armv9.0a core. However, crypto options are not enabled
  by default for Armv9 cores, so `-mcpu=neoverse-n2+crypto` is now required
  to enable crypto for this core. As far as the compiler is concerned,
  Armv9.0a has the same features enabled as Armv8.5a, with the exception
  of crypto.

Changes to the AMDGPU Backend
-----------------------------

* `llvm.sqrt.f32` is now lowered correctly. Use `llvm.amdgcn.sqrt.f32`
  for raw instruction access.

* Implemented `llvm.stacksave` and `llvm.stackrestore` intrinsics.

* Implemented :ref:`llvm.get.rounding <int_get_rounding>`

Changes to the ARM Backend
--------------------------

* Added support for Cortex-M52 CPUs.
* Added execute-only support for Armv6-M.

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------
* The code model of global variables can now be overridden by means of
  the newly added LLVM IR attribute, ``code_model``.

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* The Zfa extension version was upgraded to 1.0 and is no longer experimental.
* Zihintntl extension version was upgraded to 1.0 and is no longer experimental.
* Intrinsics were added for Zk*, Zbb, and Zbc. See https://github.com/riscv-non-isa/riscv-c-api-doc/blob/master/riscv-c-api.md#scalar-bit-manipulation-extension-intrinsics
* Default ABI with F but without D was changed to ilp32f for RV32 and to lp64f for RV64.
* The Zvbb, Zvbc, Zvkb, Zvkg, Zvkn, Zvknc, Zvkned, Zvkng, Zvknha, Zvknhb, Zvks,
  Zvksc, Zvksed, Zvksg, Zvksh, and Zvkt extension version was upgraded to 1.0
  and is no longer experimental.  However, the C intrinsics for these extensions
  are still experimental.  To use the C intrinsics for these extensions,
  ``-menable-experimental-extensions`` needs to be passed to Clang.
* XSfcie extension and SiFive CSRs and instructions that were associated with
  it have been removed. None of these CSRs and instructions were part of
  "SiFive Custom Instruction Extension" as SiFive defines it. The LLVM project
  needs to work with SiFive to define and document real extension names for
  individual CSRs and instructions.
* ``-mcpu=sifive-p450`` was added.
* CodeGen of RV32E/RV64E was supported experimentally.
* CodeGen of ilp32e/lp64e was supported experimentally.
* Support was added for the Ziccif, Ziccrse, Ziccamoa, Zicclsm, Za64rs, Za128rs
  and Zic64b extensions which were introduced as a part of the RISC-V Profiles
  specification.
* The Smepmp 1.0 extension is now supported.

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

* The LLVM filesystem class ``UniqueID`` and function ``equivalent()``
  no longer determine that distinct different path names for the same
  hard linked file actually are equal. This is an intentional tradeoff in a
  bug fix, where the bug used to cause distinct files to be considered
  equivalent on some file systems. This change fixed the issues
  https://github.com/llvm/llvm-project/issues/61401 and
  https://github.com/llvm/llvm-project/issues/22079.

Changes to the X86 Backend
--------------------------

* The ``i128`` type now matches GCC and clang's ``__int128`` type. This mainly
  benefits external projects such as Rust which aim to be binary compatible
  with C, but also fixes code generation where LLVM already assumed that the
  type matched and called into libgcc helper functions.
* Support ISA of ``USER_MSR``.
* Support ISA of ``AVX10.1-256`` and ``AVX10.1-512``.
* ``-mcpu=pantherlake`` and ``-mcpu=clearwaterforest`` are now supported.
* ``-mapxf`` is supported.
* Marking global variables with ``code_model = "small"/"large"`` in the IR now
  overrides the global code model to allow 32-bit relocations or require 64-bit
  relocations to the global variable.
* The medium code model's code generation was audited to be more similar to the
  small code model where possible.

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

* The python bindings have been removed.


Changes to the C API
--------------------

* Added ``LLVMGetTailCallKind`` and ``LLVMSetTailCallKind`` to
  allow getting and setting ``tail``, ``musttail``, and ``notail``
  attributes on call instructions.
* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstAnd``
  * ``LLVMConstOr``
  * ``LLVMConstLShr``
  * ``LLVMConstAShr``
  * ``LLVMConstZExt``
  * ``LLVMConstSExt``
  * ``LLVMConstZExtOrBitCast``
  * ``LLVMConstSExtOrBitCast``
  * ``LLVMConstIntCast``
  * ``LLVMConstFPTrunc``
  * ``LLVMConstFPExt``
  * ``LLVMConstFPToUI``
  * ``LLVMConstFPToSI``
  * ``LLVMConstUIToFP``
  * ``LLVMConstSIToFP``
  * ``LLVMConstFPCast``

* Added ``LLVMCreateTargetMachineWithOptions``, along with helper functions for
  an opaque option structure, as an alternative to ``LLVMCreateTargetMachine``.
  The option structure exposes an additional setting (i.e., the target ABI) and
  provides default values for unspecified settings.

* Added ``LLVMGetNNeg`` and ``LLVMSetNNeg`` for getting/setting the new nneg flag
  on zext instructions, and ``LLVMGetIsDisjoint`` and ``LLVMSetIsDisjoint``
  for getting/setting the new disjoint flag on or instructions.

* Added the following functions for manipulating operand bundles, as well as
  building ``call`` and ``invoke`` instructions that use operand bundles:

  * ``LLVMBuildCallWithOperandBundles``
  * ``LLVMBuildInvokeWithOperandBundles``
  * ``LLVMCreateOperandBundle``
  * ``LLVMDisposeOperandBundle``
  * ``LLVMGetNumOperandBundles``
  * ``LLVMGetOperandBundleAtIndex``
  * ``LLVMGetNumOperandBundleArgs``
  * ``LLVMGetOperandBundleArgAtIndex``
  * ``LLVMGetOperandBundleTag``

* Added ``LLVMGetFastMathFlags`` and ``LLVMSetFastMathFlags`` for getting/setting
  the fast-math flags of an instruction, as well as ``LLVMCanValueUseFastMathFlags``
  for checking if an instruction can use such flags

Changes to the CodeGen infrastructure
-------------------------------------

* A new debug type ``isel-dump`` is added to show only the SelectionDAG dumps
  after each ISel phase (i.e. ``-debug-only=isel-dump``). This new debug type
  can be filtered by function names using ``-filter-print-funcs=<function names>``,
  the same flag used to filter IR dumps after each Pass. Note that the existing
  ``-debug-only=isel`` will take precedence over the new behavior and
  print SelectionDAG dumps of every single function regardless of
  ``-filter-print-funcs``'s values.

* ``PrologEpilogInserter`` no longer supports register scavenging
  during forwards frame index elimination. Targets should use
  backwards frame index elimination instead.

* ``RegScavenger`` no longer supports forwards register
  scavenging. Clients should use backwards register scavenging
  instead, which is preferred because it does not depend on accurate
  kill flags.

Changes to the Metadata Info
---------------------------------
* Added a new loop metadata `!{!"llvm.loop.align", i32 64}`

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* llvm-symbolizer now treats invalid input as an address for which source
  information is not found.
* llvm-readelf now supports ``--extra-sym-info`` (``-X``) to display extra
  information (section name) when showing symbols.

* ``llvm-nm`` now supports the ``--line-numbers`` (``-l``) option to use
  debugging information to print symbols' filenames and line numbers.

* llvm-symbolizer and llvm-addr2line now support addresses specified as symbol names.

* llvm-objcopy now supports ``--gap-fill`` and ``--pad-to`` options, for
  ELF input and binary output files only.

Changes to LLDB
---------------------------------

* ``SBWatchpoint::GetHardwareIndex`` is deprecated and now returns -1
  to indicate the index is unavailable.
* Methods in SBHostOS related to threads have had their implementations
  removed. These methods will return a value indicating failure.
* ``SBType::FindDirectNestedType`` function is added. It's useful
  for formatters to quickly find directly nested type when it's known
  where to search for it, avoiding more expensive global search via
  ``SBTarget::FindFirstType``.
* ``lldb-vscode`` was renamed to ``lldb-dap`` and and its installation
  instructions have been updated to reflect this. The underlying functionality
  remains unchanged.
* The ``mte_ctrl`` register can now be read from AArch64 Linux core files.
* LLDB on AArch64 Linux now supports debugging the Scalable Matrix Extension
  (SME) and Scalable Matrix Extension 2 (SME2) for both live processes and core
  files. For details refer to the
  `AArch64 Linux documentation <https://lldb.llvm.org/use/aarch64-linux.html>`_.
* LLDB now supports symbol and binary acquisition automatically using the
  DEBUFINFOD protocol. The standard mechanism of specifying DEBUFINOD servers in
  the ``DEBUGINFOD_URLS`` environment variable is used by default. In addition,
  users can specify servers to request symbols from using the LLDB setting
  ``plugin.symbol-locator.debuginfod.server_urls``, override or adding to the
  environment variable.


* When running on AArch64 Linux, ``lldb-server`` now provides register
  field information for the following registers: ``cpsr``, ``fpcr``,
  ``fpsr``, ``svcr`` and ``mte_ctrl``. ::

    (lldb) register read cpsr
          cpsr = 0x80001000
               = (N = 1, Z = 0, C = 0, V = 0, SS = 0, IL = 0, <...>

  This is only available when ``lldb`` is built with XML support.
  Where possible the CPU's capabilities are used to decide which
  fields are present, however this is not always possible or entirely
  accurate. If in doubt, refer to the numerical value.

Changes to Sanitizers
---------------------
* HWASan now defaults to detecting use-after-scope bugs.

Other Changes
-------------

* The ``Flags`` field of ``llvm::opt::Option`` has been split into ``Flags``
  and ``Visibility`` to simplify option sharing between various drivers (such
  as ``clang``, ``clang-cl``, or ``flang``) that rely on Clang's Options.td.
  Overloads of ``llvm::opt::OptTable`` that use ``FlagsToInclude`` have been
  deprecated. There is a script and instructions on how to resolve conflicts -
  see https://reviews.llvm.org/D157150 and https://reviews.llvm.org/D157151 for
  details.

* On Linux, FreeBSD, and NetBSD, setting the environment variable
  ``LLVM_ENABLE_SYMBOLIZER_MARKUP`` causes tools to print stacktraces using
  :doc:`Symbolizer Markup <SymbolizerMarkupFormat>`.
  This works even if the tools have no embedded symbol information (i.e. are
  fully stripped); :doc:`llvm-symbolizer <CommandGuide/llvm-symbolizer>` can
  symbolize the markup afterwards using ``debuginfod``.

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
