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

*  The ``readnone`` calls which are crossing suspend points in coroutines will
   not be merged. Since ``readnone`` calls may access thread id and thread id
   is not a constant in coroutines. This decision may cause unnecessary
   performance regressions and we plan to fix it in later versions.

* ...

Update on required toolchains to build LLVM
-------------------------------------------

LLVM is now built with C++17 by default. This means C++17 can be used in
the code base.

The previous "soft" toolchain requirements have now been changed to "hard".
This means that the the following versions are now required to build LLVM
and there is no way to suppress this error.

* GCC >= 7.1
* Clang >= 5.0
* Apple Clang >= 10.0
* Visual Studio 2019 >= 16.7

With LLVM 16.x we will raise the version requirement of CMake used to build
LLVM. The new requirements are as follows:

* CMake >= 3.20.0

In LLVM 16.x this requirement will be "soft", there will only be a diagnostic.

With the release of LLVM 17.x this requirement will be hard and LLVM developers
can start using CMake 3.20.0 features, making it impossible to build with older
versions of CMake.

Changes to the LLVM IR
----------------------

* The ``readnone``, ``readonly``, ``writeonly``, ``argmemonly``,
  ``inaccessiblememonly`` and ``inaccessiblemem_or_argmemonly`` function
  attributes have been replaced by a single ``memory(...)`` attribute. The
  old attributes may be mapped to the new one as follows:

  * ``readnone`` -> ``memory(none)``
  * ``readonly`` -> ``memory(read)``
  * ``writeonly`` -> ``memory(write)``
  * ``argmemonly`` -> ``memory(argmem: readwrite)``
  * ``argmemonly readonly`` -> ``memory(argmem: read)``
  * ``argmemonly writeonly`` -> ``memory(argmem: write)``
  * ``inaccessiblememonly`` -> ``memory(inaccessiblemem: readwrite)``
  * ``inaccessiblememonly readonly`` -> ``memory(inaccessiblemem: read)``
  * ``inaccessiblememonly writeonly`` -> ``memory(inaccessiblemem: write)``
  * ``inaccessiblemem_or_argmemonly`` ->
    ``memory(argmem: readwrite, inaccessiblemem: readwrite)``
  * ``inaccessiblemem_or_argmemonly readonly`` ->
    ``memory(argmem: read, inaccessiblemem: read)``
  * ``inaccessiblemem_or_argmemonly writeonly`` ->
    ``memory(argmem: write, inaccessiblemem: write)``

* The constant expression variants of the following instructions has been
  removed:

  * ``fneg``

* Target extension types have been added, which allow targets to have
  types that need to be preserved through the optimizer, but otherwise are not
  introspectable by target-independent optimizations.

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

* Added support for the Cortex-A715 CPU.
* Added support for the Cortex-X3 CPU.
* Added support for the Neoverse V2 CPU.
* Added support for assembly for RME MEC (Memory Encryption Contexts).

Changes to the AMDGPU Backend
-----------------------------

Changes to the ARM Backend
--------------------------

* Support for targeting armv2, armv2A, armv3 and armv3M has been removed.
  LLVM did not, and was not ever likely to generate correct code for those
  architecture versions so their presence was misleading.

Changes to the AVR Backend
--------------------------

* ...

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

* ...

Changes to the MIPS Backend
---------------------------

* ...

Changes to the PowerPC Backend
------------------------------

* ...

Changes to the RISC-V Backend
-----------------------------

* Support for the unratified Zbe, Zbf, Zbm, Zbp, Zbr, and Zbt extensions have
  been removed.
* i32 is now a native type in the datalayout string. This enables
  LoopStrengthReduce for loops with i32 induction variables, among other
  optimizations.

Changes to the WebAssembly Backend
----------------------------------

* ...

Changes to the Windows Target
-----------------------------

* For MinGW, generate embedded ``-exclude-symbols:`` directives for symbols
  with hidden visibility, omitting them from automatic export of all symbols.
  This roughly makes hidden visibility work like it does for other object
  file formats.

* When using multi-threaded LLVM tools (such as LLD) on a Windows host with a
  large number of processors or CPU sockets, previously the LLVM ThreadPool
  would span out threads to use all processors.
  Starting with Windows Server 2022 and Windows 11, the behavior has changed,
  the OS now spans out threads automatically to all processors. This also fixes
  an affinity mask issue.
  (`D138747 <https://reviews.llvm.org/D138747>`_)

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

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstFNeg``


* The following deprecated functions have been removed, because they are
  incompatible with opaque pointers. Use the new functions accepting a separate
  function/element type instead.

  * ``LLVMBuildLoad`` -> ``LLVMBuildLoad2``
  * ``LLVMBuildCall`` -> ``LLVMBuildCall2``
  * ``LLVMBuildInvoke`` -> ``LLVMBuildInvoke2``
  * ``LLVMBuildGEP`` -> ``LLVMBuildGEP2``
  * ``LLVMBuildInBoundsGEP`` -> ``LLVMBuildInBoundsGEP2``
  * ``LLVMBuildStructGEP`` -> ``LLVMBuildStructGEP2``
  * ``LLVMBuildPtrDiff`` -> ``LLVMBuildPtrDiff2``
  * ``LLVMConstGEP`` -> ``LLVMConstGEP2``
  * ``LLVMConstInBoundsGEP`` -> ``LLVMConstInBoundsGEP2``
  * ``LLVMAddAlias`` -> ``LLVMAddAlias2``

Changes to the FastISel infrastructure
--------------------------------------

* ...

Changes to the DAG infrastructure
---------------------------------


Changes to the Metadata Info
---------------------------------

* Add Module Flags Metadata ``stack-protector-guard-symbol`` which specify a
  symbol for addressing the stack-protector guard.

Changes to the Debug Info
---------------------------------

Previously when emitting DWARF v4 and tuning for GDB, llc would use DWARF v2's
``DW_AT_bit_offset`` and ``DW_AT_data_member_location``. llc now uses DWARF v4's
``DW_AT_data_bit_offset`` regardless of tuning.

Support for ``DW_AT_data_bit_offset`` was added in GDB 8.0. For earlier versions,
you can use llc's ``-dwarf-version=3`` option to emit compatible DWARF.

Changes to the LLVM tools
---------------------------------

* ``llvm-readobj --elf-output-style=JSON`` no longer prefixes each JSON object
  with the file name. Previously, each object file's output looked like
  ``"main.o":{"FileSummary":{"File":"main.o"},...}`` but is now
  ``{"FileSummary":{"File":"main.o"},...}``. This allows each JSON object to be
  parsed in the same way, since each object no longer has a unique key. Tools
  that consume ``llvm-readobj``'s JSON output should update their parsers
  accordingly.

* ``llvm-objdump`` now uses ``--print-imm-hex`` by default, which brings its
  default behavior closer in line with ``objdump``.

Changes to LLDB
---------------------------------

Changes to Sanitizers
---------------------


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
