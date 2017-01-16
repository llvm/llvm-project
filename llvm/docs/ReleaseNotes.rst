========================
LLVM 4.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 4.0.0 release.  You may
   prefer the `LLVM 3.9 Release Notes <http://llvm.org/releases/3.9.0/docs
   /ReleaseNotes.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 4.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Non-comprehensive list of changes in this release
=================================================
* The C API functions LLVMAddFunctionAttr, LLVMGetFunctionAttr,
  LLVMRemoveFunctionAttr, LLVMAddAttribute, LLVMRemoveAttribute,
  LLVMGetAttribute, LLVMAddInstrAttribute and
  LLVMRemoveInstrAttribute have been removed.

* The C API enum LLVMAttribute has been deleted.

.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* The definition and uses of LLVM_ATRIBUTE_UNUSED_RESULT in the LLVM source
  were replaced with LLVM_NODISCARD, which matches the C++17 [[nodiscard]]
  semantics rather than gcc's __attribute__((warn_unused_result)).

* Minimum compiler version to build has been raised to GCC 4.8 and VS 2015.

* The Timer related APIs now expect a Name and Description. When upgrading code
  the previously used names should become descriptions and a short name in the
  style of a programming language identifier should be added.

* LLVM now handles invariant.group across different basic blocks, which makes
  it possible to devirtualize virtual calls inside loops.

* ... next change ...

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

   Improvements to ThinLTO (-flto=thin)
   ------------------------------------
   * Integration with profile data (PGO). When available, profile data 
     enables more accurate function importing decisions, as well as 
     cross-module indirect call promotion.
   * Significant build-time and binary-size improvements when compiling with 
     debug info (-g).

Changes to the LLVM IR
----------------------

Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

 During this release ...


Changes to the PowerPC Target
-----------------------------

 During this release ...

Changes to the X86 Target
-------------------------

 During this release ...

Changes to the AMDGPU Target
-----------------------------

 During this release ...

Changes to the AVR Target
-----------------------------

* The entire backend has been merged in-tree with all tests passing. All of
  the instruction selection code and the machine code backend has landed
  recently and is fully usable.

Changes to the OCaml bindings
-----------------------------

* The attribute API was completely overhauled, following the changes
  to the C API.


External Open Source Projects Using LLVM 4.0.0
==============================================

* A project...


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<http://llvm.org/>`_, in particular in the `documentation
<http://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <http://llvm.org/docs/#maillist>`_.
