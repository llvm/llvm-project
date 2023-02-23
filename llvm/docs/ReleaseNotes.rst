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

* ...

Changes to the RISC-V Backend
-----------------------------

* Assembler support for version 1.0.1 of the Zcb extension was added.
* Zca, Zcf, and Zcd extensions were upgraded to version 1.0.1.
* vsetvli intrinsics no longer have side effects. They may now be combined,
  moved, deleted, etc. by optimizations.
* Adds support for the vendor-defined XTHeadBa (address-generation) extension.
* Adds support for the vendor-defined XTHeadBb (basic bit-manipulation) extension.
* Adds support for the vendor-defined XTHeadBs (single-bit) extension.
* Adds support for the vendor-defined XTHeadMac (multiply-accumulate instructions) extension.
* Added support for the vendor-defined XTHeadMemPair (two-GPR memory operations)
  extension disassembler/assembler.
* Support for the now-ratified Zawrs extension is no longer experimental.
* Adds support for the vendor-defined XTHeadCmo (cache management operations) extension.
* Adds support for the vendor-defined XTHeadSync (multi-core synchronization instructions) extension.

Changes to the WebAssembly Backend
----------------------------------

* ...

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* ``LLVMContextSetOpaquePointers``, a temporary API to pin to legacy typed
  pointer, has been removed.

Changes to the FastISel infrastructure
--------------------------------------

* ...

Changes to the DAG infrastructure
---------------------------------


Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

* The DWARFv5 feature of attaching `DW_AT_default_value` to defaulted template
  parameters will now be available in any non-strict DWARF mode and in a wider
  range of cases than previously. (`D139953 <https://reviews.llvm.org/D139953>`_, `D139988 <https://reviews.llvm.org/D139988>`_)

* The `DW_AT_name` on `DW_AT_typedef`s for alias templates will now omit defaulted
  template parameters. (`D142268 <https://reviews.llvm.org/D142268>`_)

Changes to the LLVM tools
---------------------------------

Changes to LLDB
---------------------------------

* In the results of commands such as `expr` and `frame var`, type summaries will now
  omit defaulted template parameters. The full template parameter list can still be
  viewed with `expr --raw-output`/`frame var --raw-output`. (`D141828 <https://reviews.llvm.org/D141828>`_)

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
