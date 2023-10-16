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

* Added `llvm.exp10` intrinsic.

Changes to LLVM infrastructure
------------------------------

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

Changes to the AMDGPU Backend
-----------------------------

* `llvm.sqrt.f32` is now lowered correctly. Use `llvm.amdgcn.sqrt.f32`
  for raw instruction access.

* Implemented `llvm.stacksave` and `llvm.stackrestore` intrinsics.

* Implemented :ref:`llvm.get.rounding <int_get_rounding>`

Changes to the ARM Backend
--------------------------

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* The Zfa extension version was upgraded to 1.0 and is no longer experimental.
* Zihintntl extension version was upgraded to 1.0 and is no longer experimental.
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
* Support for the unratified Zbe, Zbf, Zbm, Zbp, Zbr, and Zbt extensions have
  been removed.
* i32 is now a native type in the datalayout string. This enables
  LoopStrengthReduce for loops with i32 induction variables, among other
  optimizations.

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

* Add support for the ``RDMSRLIST and WRMSRLIST`` instructions.
* Add support for the ``WRMSRNS`` instruction.
* Support ISA of ``AMX-FP16`` which contains ``tdpfp16ps`` instruction.
* Support ISA of ``CMPCCXADD``.
* Support ISA of ``AVX-IFMA``.
* Support ISA of ``AVX-VNNI-INT8``.
* Support ISA of ``AVX-NE-CONVERT``.
* ``-mcpu=raptorlake``, ``-mcpu=meteorlake`` and ``-mcpu=emeraldrapids`` are now supported.
* ``-mcpu=sierraforest``, ``-mcpu=graniterapids`` and ``-mcpu=grandridge`` are now supported.

* ``__builtin_unpredictable`` (unpredictable metadata in LLVM IR), is handled by X86 Backend.
  ``X86CmovConversion`` pass now respects this builtin and does not convert CMOVs to branches.
* Add support for the ``PBNDKB`` instruction.

* Support ISA of ``SHA512``.
* Support ISA of ``SM3``.
* Support ISA of ``SM4``.
* Support ISA of ``AVX-VNNI-INT16``.
* ``-mcpu=graniterapids-d`` is now supported.

* The ``i128`` type now matches GCC and clang's ``__int128`` type. This mainly
  benefits external projects such as Rust which aim to be binary compatible
  with C, but also fixes code generation where LLVM already assumed that the
  type matched and called into libgcc helper functions.
* Support ISA of ``USER_MSR``.

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

Changes to the CodeGen infrastructure
-------------------------------------

* ``PrologEpilogInserter`` no longer supports register scavenging
  during forwards frame index elimination. Targets should use
  backwards frame index elimination instead.

* ``RegScavenger`` no longer supports forwards register
  scavenging. Clients should use backwards register scavenging
  instead, which is preferred because it does not depend on accurate
  kill flags.

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

* llvm-symbolizer now treats invalid input as an address for which source
  information is not found.
* llvm-readelf now supports ``--extra-sym-info`` (``-X``) to display extra
  information (section name) when showing symbols.

* ``llvm-readobj --elf-output-style=JSON`` no longer prefixes each JSON object
  with the file name. Previously, each object file's output looked like
  ``"main.o":{"FileSummary":{"File":"main.o"},...}`` but is now
  ``{"FileSummary":{"File":"main.o"},...}``. This allows each JSON object to be
  parsed in the same way, since each object no longer has a unique key. Tools
  that consume ``llvm-readobj``'s JSON output should update their parsers
  accordingly.

* ``llvm-objdump`` now uses ``--print-imm-hex`` by default, which brings its
  default behavior closer in line with ``objdump``.
* ``llvm-nm`` now supports the ``--line-numbers`` (``-l``) option to use
  debugging information to print symbols' filenames and line numbers.

Changes to LLDB
---------------------------------

* Methods in SBHostOS related to threads have had their implementations
  removed. These methods will return a value indicating failure.
* ``SBType::FindDirectNestedType`` function is added. It's useful
  for formatters to quickly find directly nested type when it's known
  where to search for it, avoiding more expensive global search via
  ``SBTarget::FindFirstType``.

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
