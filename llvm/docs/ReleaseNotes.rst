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

* Typed pointers are no longer supported. See the `opaque pointers
  <OpaquePointers.html>`__ documentation for migration instructions.

* The ``nofpclass`` attribute was introduced. This allows more
  optimizations around special floating point value comparisons.

* The constant expression variants of the following instructions have been
  removed:

  * ``select``

Changes to LLVM infrastructure
------------------------------

* The legacy optimization pipeline has been removed.

* Alloca merging in the inliner has been removed, since it only worked with the
  legacy inliner pass. Backend stack coloring should handle cases alloca
  merging initially set out to handle.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* Added Assembly Support for the 2022 A-profile extensions FEAT_GCS (Guarded
  Control Stacks), FEAT_CHK (Check Feature Status), and FEAT_ATS1A.

Changes to the AMDGPU Backend
-----------------------------
* More fine-grained synchronization around barriers for newer architectures
  (gfx90a+, gfx10+). The AMDGPU backend now omits previously automatically
  generated waitcnt instructions before barriers, allowing for more precise
  control. Users must now use memory fences to implement fine-grained
  synchronization strategies around barriers. Refer to `AMDGPU memory model
  <AMDGPUUsage.html#memory-model>`__.

Changes to the ARM Backend
--------------------------

- The hard-float ABI is now available in Armv8.1-M configurations that
  have integer MVE instructions (and therefore have FP registers) but
  no scalar or vector floating point computation.

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

Changes to the MIPS Backend
---------------------------

* ...

Changes to the PowerPC Backend
------------------------------

* A new option ``-mxcoff-roptr`` is added to ``clang`` and ``llc``. When this
  option is present, constant objects with relocatable address values are put
  into the RO data section. This option should be used with the ``-fdata-sections``
  option, and is not supported with ``-fno-data-sections``. The option is
  only supported on AIX.
* On AIX, teach the profile runtime to check for a build-id string; such string
  can be created by the -mxcoff-build-id option.

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

Changes to the WebAssembly Backend
----------------------------------

* ...

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------
* Support ISA of ``AVX-IFMA``.

* Add support for the ``RDMSRLIST and WRMSRLIST`` instructions.
* Add support for the ``WRMSRNS`` instruction.
* Support ISA of ``AMX-FP16`` which contains ``tdpfp16ps`` instruction.
* Support ISA of ``CMPCCXADD``.
* Support ISA of ``AVX-VNNI-INT8``.
* Support ISA of ``AVX-NE-CONVERT``.
* ``-mcpu=raptorlake``, ``-mcpu=meteorlake`` and ``-mcpu=emeraldrapids`` are now supported.
* ``-mcpu=sierraforest``, ``-mcpu=graniterapids`` and ``-mcpu=grandridge`` are now supported.

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* ``LLVMContextSetOpaquePointers``, a temporary API to pin to legacy typed
  pointer, has been removed.
* Functions for adding legacy passes like ``LLVMAddInstructionCombiningPass``
  have been removed.
* Removed ``LLVMPassManagerBuilderRef`` and functions interacting with it.
  These belonged to the no longer supported legacy pass manager.
* As part of the opaque pointer transition, ``LLVMGetElementType`` no longer
  gives the pointee type of a pointer type.
* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstSelect``

Changes to the FastISel infrastructure
--------------------------------------

* ...

Changes to the DAG infrastructure
---------------------------------


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

Other Changes
-------------

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
