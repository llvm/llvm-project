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

* ...

Update on required toolchains to build LLVM
-------------------

With LLVM 15.x we will raise the version requirements of the toolchain used
to build LLVM. The new requirements are as follows:

* GCC >= 7.1
* Clang >= 5.0
* Apple Clang >= 9.3
* Visual Studio 2019 >= 16.7

In LLVM 15.x these requirements will be "soft" requirements and the version
check can be skipped by passing -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON
to CMake.

With the release of LLVM 16.x these requirements will be hard and LLVM developers
can start using C++17 features, making it impossible to build with older
versions of these toolchains.

Changes to the LLVM IR
----------------------

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

Changes to the AMDGPU Backend
-----------------------------

* ...

Changes to the ARM Backend
--------------------------

* Added support for the Armv9-A, Armv9.1-A and Armv9.2-A architectures.
* Added support for the Armv8.1-M PACBTI-M extension.
* Added support for the Armv9-A, Armv9.1-A and Armv9.2-A architectures.
* Added support for the Armv8.1-M PACBTI-M extension.
* Removed the deprecation of ARMv8-A T32 Complex IT blocks. No deprecation
  warnings will be generated and -mrestrict-it is now always off by default.
  Previously it was on by default for Armv8 and off for all other architecture
  versions.

Changes to the AVR Backend
--------------------------

* ...

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

* The Zvfh extension was added.

Changes to the WebAssembly Backend
----------------------------------

* ...

Changes to the X86 Backend
--------------------------

* ...

Changes to the OCaml bindings
-----------------------------


Changes to the C API
--------------------

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

Changes to LLDB
---------------------------------

Changes to Sanitizers
---------------------

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
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
