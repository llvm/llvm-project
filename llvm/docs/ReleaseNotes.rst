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

* The minimum Python version has been raised from 3.6 to 3.8 across all of LLVM.
  This enables the use of many new Python features, aligning more closely with
  modern Python best practices, and improves CI maintainability
  See `#78828 <https://github.com/llvm/llvm-project/pull/78828>`_ for more info.

Changes to the LLVM IR
----------------------

* Added Memory Model Relaxation Annotations (MMRAs).
* Added ``nusw`` and ``nuw`` flags to ``getelementptr`` instruction.
* Renamed ``llvm.experimental.vector.reverse`` intrinsic to ``llvm.vector.reverse``.
* Renamed ``llvm.experimental.vector.splice`` intrinsic to ``llvm.vector.splice``.
* Renamed ``llvm.experimental.vector.interleave2`` intrinsic to ``llvm.vector.interleave2``.
* Renamed ``llvm.experimental.vector.deinterleave2`` intrinsic to ``llvm.vector.deinterleave2``.
* The constant expression variants of the following instructions have been
  removed:

  * ``icmp``
  * ``fcmp``
* LLVM has switched from using debug intrinsics in textual IR to using debug
  records by default. Details of the change and instructions on how to update
  any downstream tools and tests can be found in the `migration docs
  <https://llvm.org/docs/RemoveDIsDebugInfo.html>`_.
* Semantics of MC/DC intrinsics have been changed.

  * ``llvm.instprof.mcdc.parameters``: 3rd argument has been changed
    from bytes to bits.
  * ``llvm.instprof.mcdc.condbitmap.update``: Removed.
  * ``llvm.instprof.mcdc.tvbitmap.update``: 3rd argument has been
    removed. The next argument has been changed from byte index to bit
    index.

Changes to LLVM infrastructure
------------------------------

Changes to building LLVM
------------------------

- The ``LLVM_ENABLE_TERMINFO`` flag has been removed. LLVM no longer depends on
  terminfo and now always uses the ``TERM`` environment variable for color
  support autodetection.

Changes to TableGen
-------------------

- We can define type aliases via new keyword ``deftype``.

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* Added support for Cortex-R82AE, Cortex-A78AE, Cortex-A520AE, Cortex-A720AE,
  Cortex-A725, Cortex-X925, Neoverse-N3, Neoverse-V3 and Neoverse-V3AE CPUs.

* ``-mbranch-protection=standard`` now enables FEAT_PAuth_LR by
  default when the feature is enabled. The new behaviour results 
  in ``standard`` being equal to ``bti+pac-ret+pc`` when ``+pauth-lr``
  is passed as part of ``-mcpu=`` options.

Changes to the AMDGPU Backend
-----------------------------

* Implemented the ``llvm.get.fpenv`` and ``llvm.set.fpenv`` intrinsics.

* Implemented :ref:`llvm.get.rounding <int_get_rounding>` and :ref:`llvm.set.rounding <int_set_rounding>`

Changes to the ARM Backend
--------------------------

* Added support for Cortex-R52+ CPU.
* FEAT_F32MM is no longer activated by default when using `+sve` on v8.6-A or greater. The feature is still available and can be used by adding `+f32mm` to the command line options.
* armv8-r now implies only fp-armv8d16sp, rather than neon and full fp-armv8. These features are still included by default for cortex-r52. The default cpu for armv8-r is now "generic", for compatibility with variants that do not include neon, fp64, and d32.

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------

* i32 is now a native type in the datalayout string. This enables
  LoopStrengthReduce for loops with i32 induction variables, among other
  optimizations.

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* Added full support for the experimental Zabha (Byte and
  Halfword Atomic Memory Operations) extension.
* Added assembler/disassembler support for the experimenatl Zalasr
  (Load-Acquire and Store-Release) extension.
* The names of the majority of the S-prefixed (supervisor-level) extension
  names in the RISC-V profiles specification are now recognised.
* Codegen support was added for the Zimop (May-Be-Operations) extension.
* The experimental Ssnpm, Smnpm, Smmpm, Sspm, and Supm 0.8.1 Pointer Masking extensions are supported.
* The experimental Ssqosid extension is supported.
* Zacas is no longer experimental.
* Added the CSR names from the Resumable Non-Maskable Interrupts (Smrnmi) extension.
* llvm-objdump now prints disassembled opcode bytes in groups of 2 or 4 bytes to
  match GNU objdump. The bytes within the groups are in big endian order.
* Added smstateen extension to -march. CSR names for smstateen were already supported.
* Zaamo and Zalrsc are no longer experimental.
* Processors that enable post reg-alloc scheduling (PostMachineScheduler) by default should use the `UsePostRAScheduler` subtarget feature. Setting `PostRAScheduler = 1` in the scheduler model will have no effect on the enabling of the PostMachineScheduler.
* Zabha is no longer experimental.
* B (the collection of the Zba, Zbb, Zbs extensions) is supported.
* Added smcdeleg, ssccfg, smcsrind, and sscsrind extensions to -march.

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

- Removed knl/knm specific ISA intrinsics: AVX512PF, AVX512ER, PREFETCHWT1,
  while assembly encoding/decoding supports are kept.

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

* Added ``LLVMGetBlockAddressFunction`` and ``LLVMGetBlockAddressBasicBlock``
  functions for accessing the values in a blockaddress constant.

* Added ``LLVMConstStringInContext2`` function, which better matches the C++
  API by using ``size_t`` for string length. Deprecated ``LLVMConstStringInContext``.

* Added the following functions for accessing a function's prefix data:

  * ``LLVMHasPrefixData``
  * ``LLVMGetPrefixData``
  * ``LLVMSetPrefixData``

* Added the following functions for accessing a function's prologue data:

  * ``LLVMHasPrologueData``
  * ``LLVMGetPrologueData``
  * ``LLVMSetPrologueData``

* Deprecated ``LLVMConstNUWNeg`` and ``LLVMBuildNUWNeg``.

* Added ``LLVMAtomicRMWBinOpUIncWrap`` and ``LLVMAtomicRMWBinOpUDecWrap`` to
  ``LLVMAtomicRMWBinOp`` enum for AtomicRMW instructions.

* Added ``LLVMCreateConstantRangeAttribute`` function for creating ConstantRange Attributes.

* Added the following functions for creating and accessing data for CallBr instructions:

  * ``LLVMBuildCallBr``
  * ``LLVMGetCallBrDefaultDest``
  * ``LLVMGetCallBrNumIndirectDests``
  * ``LLVMGetCallBrIndirectDest``

* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstICmp``
  * ``LLVMConstFCmp``

**Note:** The following changes are due to the removal of the debug info
intrinsics from LLVM and to the introduction of debug records into LLVM.
They are described in detail in the `debug info migration guide <https://llvm.org/docs/RemoveDIsDebugInfo.html>`_.

* Added the following functions to insert before the indicated instruction but
  after any attached debug records.

  * ``LLVMPositionBuilderBeforeDbgRecords``
  * ``LLVMPositionBuilderBeforeInstrAndDbgRecords``

  Same as ``LLVMPositionBuilder`` and ``LLVMPositionBuilderBefore`` except the
  insertion position is set to before the debug records that precede the target
  instruction. ``LLVMPositionBuilder`` and ``LLVMPositionBuilderBefore`` are
  unchanged.

* Added the following functions to get/set the new non-instruction debug info format.
  They will be deprecated in the future and they are just a transition aid.

  * ``LLVMIsNewDbgInfoFormat``
  * ``LLVMSetIsNewDbgInfoFormat``

* Added the following functions to insert a debug record (new debug info format).

  * ``LLVMDIBuilderInsertDeclareRecordBefore``
  * ``LLVMDIBuilderInsertDeclareRecordAtEnd``
  * ``LLVMDIBuilderInsertDbgValueRecordBefore``
  * ``LLVMDIBuilderInsertDbgValueRecordAtEnd``

* Deleted the following functions that inserted a debug intrinsic (old debug info format).

  * ``LLVMDIBuilderInsertDeclareBefore``
  * ``LLVMDIBuilderInsertDeclareAtEnd``
  * ``LLVMDIBuilderInsertDbgValueBefore``
  * ``LLVMDIBuilderInsertDbgValueAtEnd``

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

* LLVM has switched from using debug intrinsics internally to using debug
  records by default. This should happen transparently when using the DIBuilder
  to construct debug variable information, but will require changes for any code
  that interacts with debug intrinsics directly. Debug intrinsics will only be
  supported on a best-effort basis from here onwards; for more information, see
  the `migration docs <https://llvm.org/docs/RemoveDIsDebugInfo.html>`_.

Changes to the LLVM tools
---------------------------------
* llvm-nm and llvm-objdump can now print symbol information from linked
  WebAssembly binaries, using information from exports or the "name"
  section for functions, globals and data segments. Symbol addresses and sizes
  are printed as offsets in the file, allowing for binary size analysis. Wasm
  files using reference types and GC are also supported (but also only for
  functions, globals, and data, and only for listing symbols and names).

* llvm-ar now utilizes LLVM_DEFAULT_TARGET_TRIPLE to determine the archive format
  if it's not specified with the ``--format`` argument and cannot be inferred from
  input files.

* llvm-ar now allows specifying COFF archive format with ``--format`` argument
  and uses it by default for COFF targets.

* llvm-ranlib now supports ``-V`` as an alias for ``--version``.
  ``-v`` (``--verbose`` in llvm-ar) has been removed.
  (`#87661 <https://github.com/llvm/llvm-project/pull/87661>`_)

* llvm-objcopy now supports ``--set-symbol-visibility`` and
  ``--set-symbols-visibility`` options for ELF input to change the
  visibility of symbols.

* llvm-objcopy now supports ``--skip-symbol`` and ``--skip-symbols`` options
  for ELF input to skip the specified symbols when executing other options
  that can change a symbol's name, binding or visibility.

* llvm-objcopy now supports ``--compress-sections`` to compress or decompress
  arbitrary sections not within a segment.
  (`#85036 <https://github.com/llvm/llvm-project/pull/85036>`_.)

* llvm-profgen now supports COFF+DWARF binaries. This enables Sample-based PGO
  on Windows using Intel VTune's SEP. For details on usage, see the `end-user
  documentation for SPGO
  <https://clang.llvm.org/docs/UsersManual.html#using-sampling-profilers>`_.

* llvm-readelf's ``-r`` output for RELR has been improved.
  (`#89162 <https://github.com/llvm/llvm-project/pull/89162>`_)
  ``--raw-relr`` has been removed.

* llvm-mca now aborts by default if it is given bad input where previously it
  would continue. Additionally, it can now continue when it encounters
  instructions which lack scheduling information. The behaviour can be
  controlled by the newly introduced
  `--skip-unsupported-instructions=<none|lack-sched|parse-failure|any>`, as
  documented in `--help` output and the command guide. (`#90474
  <https://github.com/llvm/llvm-project/pull/90474>`)

* llvm-readobj's LLVM output format for ELF core files has been changed.
  Similarly, the JSON format has been fixed for this case. The NT_FILE note
  now has a map for the mapped files. (`#92835
  <https://github.com/llvm/llvm-project/pull/92835>`).

* llvm-cov now generates HTML report with JavaScript code to allow simple
  jumping between uncovered parts (lines/regions/branches) of code 
  using buttons on top-right corner of the page or using keys (L/R/B or 
  jumping in reverse direction with shift+L/R/B). (`#95662
  <https://github.com/llvm/llvm-project/pull/95662>`).

Changes to LLDB
---------------------------------

Changes to Sanitizers
---------------------

Other Changes
-------------

External Open Source Projects Using LLVM 19
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
