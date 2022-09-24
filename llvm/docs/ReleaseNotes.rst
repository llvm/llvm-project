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

LLVM is now built with C++17 by default. This means C++17 can be used in
the code base.

The previous "soft" toolchain requirements have now been changed to "hard".
This means that the the following versions are now required to build LLVM
and there is no way to suppress this error.

* GCC >= 7.1
* Clang >= 5.0
* Apple Clang >= 9.3
* Visual Studio 2019 >= 16.7

Changes to the LLVM IR
----------------------

* The constant expression variants of the following instructions has been
  removed:

  * ``fneg``

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

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

Changes to the WebAssembly Backend
----------------------------------

* ...

Changes to the Windows Target
-----------------------------

* For MinGW, generate embedded ``-exclude-symbols:`` directives for symbols
  with hidden visibility, omitting them from automatic export of all symbols.
  This roughly makes hidden visibility work like it does for other object
  file formats.

Changes to the X86 Backend
--------------------------

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

* The following functions for creating constant expressions have been removed,
  because the underlying constant expressions are no longer supported. Instead,
  an instruction should be created using the ``LLVMBuildXYZ`` APIs, which will
  constant fold the operands if possible and create an instruction otherwise:

  * ``LLVMConstFNeg``

Changes to the Go bindings
--------------------------


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

During this release ...

Changes to the LLVM tools
---------------------------------

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
