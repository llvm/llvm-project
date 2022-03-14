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
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
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


.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

* ...

Changes to the LLVM IR
----------------------

* Using the legacy pass manager for the optimization pipeline is deprecated and
  will be removed after LLVM 14. In the meantime, only minimal effort will be
  made to maintain the legacy pass manager for the optimization pipeline.
* Max allowed integer type was reduced from 2^24-1 bits to 2^23 bits.
* Max allowed alignment was increased from 2^29 to 2^32.

Changes to building LLVM
------------------------

* Building LLVM with Visual Studio now requires version 2019 or later.

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

* Added support for the Armv9-A, Armv9.1-A and Armv9.2-A architectures.
* The compiler now recognises the "tune-cpu" function attribute to support
  the use of the -mtune frontend flag. This allows certain scheduling features
  and optimisations to be enabled independently of the architecture. If the
  "tune-cpu" attribute is absent it tunes according to the "target-cpu".
* Fixed relocations against temporary symbols (e.g. in jump tables and
  constant pools) in large COFF object files.
* Auto-vectorization now targets SVE by default when available.

Changes to the ARM Backend
--------------------------

* Added support for the Armv9-A, Armv9.1-A and Armv9.2-A architectures.
* Added support for the Armv8.1-M PACBTI-M extension.
* Changed the assembly comment string for MSVC targets to ``@`` (consistent
  with the MinGW and ELF targets), freeing up ``;`` to be used as
  statement separator.

Changes to the MIPS Target
--------------------------

During this release ...

Changes to the Hexagon Target
-----------------------------

* ...

Changes to the PowerPC Target
-----------------------------

Linux improvements:

* Provided a number of builtins for compatibility with the XL compiler.
* Allow MMA builtin types in pre-P10 compilation units.
* Add support for Return Oriented Programming (ROP) protection for 32 bit.
* Refactored code to use more inclusive language.
* Switched to LLD as the default linker for pre-built Linux binaries.
* Enabled IEEE quad long double on Linux via ``PPC_LINUX_DEFAULT_IEEELONGDOUBLE``
  in cmake config.

  * Added ``__ibm128`` type to represent IBM double-double format, also available
    as ``__attribute__((mode(IF)))``.
  * ``-mfloat128`` can now be used in Linux subtargets with VSX enabled.

* Added quadword atomic load/store support in codegen; not enabled by default.
* Codegen improvements for splat load, byval parameter, stack lowering, etc.
* Implemented P10 instruction scheduling model.
* Implemented P10 instruction fusion pairs.
* Improved handling of ``#pragma clang loop unroll_and_jam``.
* Various bug fixes.

AIX Support/improvements:

* Variadic (ellipsis) functions with C complex types are now supported.
* Added toc-data support for AIX 64-bit.
* Added toc-data support for read-only globals.
* Updated default target on AIX from pwr4 to pwr7.
* AIX 64-bit code generation now uses fast-isel for O0.
* Added DWARF support for 32-bit XCOFF.

Changes to the RISC-V Target
----------------------------

* Codegen improvements for RV64 around the selection of addw/subw/mulw/slliw
  instructions and removal of redundant sext.w instructions (using the new
  RISCVSExtWRemoval pass).
* The various RISC-V vector extensions were updated to version 1.0 and are no
  longer experimental.
* The Zba, Zbb, Zbc, and Zbs bit-manipulation extensions were updated to
  version 1.0 and are no longer experimental.
* Added MC layer support for the ratified scalar cryptography extensions.
* The Zfh and Zfhmin extensions for half-precision floating point were updated
  to version 1.0 and are no longer experimental.
* Added support for the ``.insn`` directive.
* Various improvements to immediate materialisation, including when
  bit-manipulation extensions are enabled. Additionally, the constant pool is
  now used for large integers.
* Added support for constrained FP intrinsics for scalar types.
* Added support for CSRs introduced in the Sscofpmf, Smstateen, and Sstc
  extensions.
* The experimental 'Zbproposedc' extension was removed, as was the 'B'
  extension (including all bit-manipulation sub-extensions). Individual 'Zb*'
  extensions should be used instead.

Changes to the X86 Target
-------------------------

During this release ...

* Support for ``AVX512-FP16`` instructions has been added.
* Removed incomplete support for Intel MPX.
  (`D111517 <https://reviews.llvm.org/D111517>`_)

Changes to the AMDGPU Target
-----------------------------

During this release ...

Changes to the AVR Target
-----------------------------

During this release ...

Changes to the WebAssembly Target
---------------------------------

During this release ...

Changes to the Windows Target
-----------------------------

* Changed how the ``.pdata`` sections refer to the code they're describing,
  to avoid conflicting unwind info if weak symbols are overridden.

* Fixed code generation for calling support routines for converting 128 bit
  integers from/to floats on x86_64.

* The preferred path separator form (backslashes or forward slashes) can be
  configured in Windows builds of LLVM now, with the
  ``LLVM_WINDOWS_PREFER_FORWARD_SLASH`` CMake option. This defaults to
  true in MinGW builds of LLVM.

* Set proper COFF symbol types for function aliases (e.g. for Itanium C++
  constructors), making sure that GNU ld exports all of them correctly as
  functions, not data, when linking a DLL.

* Handling of temporary files on more uncommon file systems (network
  mounts, ramdisks) on Windows is fixed now (which previously either
  errored out or left stray files behind).

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* ``LLVMSetInstDebugLocation`` has been deprecated in favor of the more general
  ``LLVMAddMetadataToInst``.

* Fixed building LLVM-C.dll for i386 targets with MSVC, which had been broken
  since the LLVM 8.0.0 release.

Changes to the Go bindings
--------------------------


Changes to the FastISel infrastructure
--------------------------------------

* ...

Changes to the DAG infrastructure
---------------------------------


Changes to the Debug Info
---------------------------------

During this release ...

Changes to the LLVM tools
---------------------------------

* llvm-cov: `-name-allowlist` is now accepted in addition to `-name-whitelist`.
  `-name-whitelist` is marked as deprecated and to be removed in future
  releases.

* llvm-ar now supports ``--thin`` for creating a thin archive. The modifier
  ``T`` has a different meaning in some ar implementations.
  (`D116979 <https://reviews.llvm.org/D116979>`_)
* llvm-ar now supports reading big archives for XCOFF.
  (`D111889 <https://reviews.llvm.org/D111889>`_)
* llvm-nm now demangles Rust symbols.
  (`D111937 <https://reviews.llvm.org/D111937>`_)
* llvm-objcopy's ELF port now avoids reordering section headers to preserve ``st_shndx`` fields of dynamic symbols.
  (`D107653 <https://reviews.llvm.org/D112116>`_)
* llvm-objcopy now supports ``--update-section`` for ELF and Mach-O.
  (`D112116 <https://reviews.llvm.org/D112116>`_)
  (`D117281 <https://reviews.llvm.org/D117281>`_)
* llvm-objcopy now supports ``--subsystem`` for PE/COFF.
  (`D116556 <https://reviews.llvm.org/D116556>`_)
* llvm-objcopy now supports mips64le relocations for ELF.
  (`D115635 <https://reviews.llvm.org/D115635>`_)
* llvm-objcopy ``--rename-section`` now renames relocation sections together with their targets.
  (`D110352 <https://reviews.llvm.org/D110352>`_)
* llvm-objdump ``--symbolize-operands`` now supports PowerPC.
  (`D114492 <https://reviews.llvm.org/D114492>`_)
* llvm-objdump ``-p`` now dumps PE header.
  (`D113356 <https://reviews.llvm.org/D113356>`_)
* llvm-objdump ``-R`` now supports ELF position-dependent executables.
  (`D110595 <https://reviews.llvm.org/D110595>`_)
* llvm-objdump ``-T`` now prints symbol versions.
  (`D108097 <https://reviews.llvm.org/D108097>`_)
* llvm-readobj: Improved printing of symbols in Windows unwind data.
* llvm-readobj now supports ``--elf-output-style=JSON`` for JSON output and
  ``--pretty-print`` for pretty printing of this output.
  (`D114225 <https://reviews.llvm.org/D114225>`_)
* llvm-readobj now supports several dump styles (``--needed-libs, --relocs, --syms``) for XCOFF.
* llvm-symbolizer now supports `--debuginfod <https://llvm.org/docs/CommandGuide/llvm-symbolizer.html>`.
  (`D113717 <https://reviews.llvm.org/D113717>`_)
* ``llvm-cov`` now accepts "allowlist" spelling for ``-name-allowlist``.
* ``llvm-nm`` now supports XCOFF object files.
* Added ``--needed-libs``, aux header, and symbols support in ``llvm-readobj``.
* Added ``--symbolize-operands`` support in ``llvm-objdump``.
* Tools that read archive files now support reading AIX big format archive files.
* Added dump section support in ``obj2yaml``.
* Added ``yaml2obj`` support for 64-bit XCOFF.

Changes to LLDB
---------------------------------

* A change in Clang's type printing has changed the way LLDB names array types
  (from ``int [N]`` to ``int[N]``) - LLDB pretty printer type name matching
  code may need to be updated to handle this.
* The following commands now ignore non-address bits (e.g. AArch64 pointer
  signatures) in address arguments. In addition, non-address bits will not
  be shown in the output of the commands.

  * ``memory find``
  * ``memory read``
  * ``memory region`` (see below)
  * ``memory tag read``
  * ``memory tag write``

* The ``memory region`` command and ``GetMemoryRegionInfo`` API method now
  ignore non-address bits in the address parameter. This also means that on
  systems with non-address bits the last (usually unmapped) memory region
  will not extend to 0xF...F. Instead it will end at the end of the mappable
  range that the virtual address size allows.

* The ``memory read`` command has a new option ``--show-tags``. Use this option
  to show memory tags beside the contents of tagged memory ranges.

* Fixed continuing from breakpoints and singlestepping on Windows on ARM/ARM64.

* LLDB has been included in Windows on ARM64 binary release with Python support
  disabled.

Changes to Sanitizers
---------------------

Changes to BOLT
---------------------

* BOLT project is added to the LLVM monorepo. BOLT is a post-link optimizer
  developed to speed up large applications. Build and usage instructions are
  given in `README <https://github.com/llvm/llvm-project/tree/main/bolt>`_.

External Open Source Projects Using LLVM 14
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
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
