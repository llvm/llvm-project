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

* `llvm.sqrt.f64` is now lowered correctly. Use `llvm.amdgcn.sqrt.f64`
  for raw instruction access.

* Implemented `llvm.stacksave` and `llvm.stackrestore` intrinsics.

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

* Zihintntl extension version was upgraded to 1.0 and is no longer experimental.

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

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

Changes to LLDB
---------------------------------

* Methods in SBHostOS related to threads have had their implementations
  removed. These methods will return a value indicating failure.

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
