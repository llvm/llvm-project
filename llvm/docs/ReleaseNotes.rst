======================
LLVM 3.7 Release Notes
======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 3.7 release.  You may
   prefer the `LLVM 3.6 Release Notes <http://llvm.org/releases/3.6.0/docs
   /ReleaseNotes.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.7.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* The minimum required Visual Studio version for building LLVM is now 2013
  Update 4.

* A new documentation page, :doc:`Frontend/PerformanceTips`, contains a
  collection of tips for frontend authors on how to generate IR which LLVM is
  able to effectively optimize.

* The DataLayout is no longer optional. All the IR level optimizations expects
  it to be present and the API has been changed to use a reference instead of
  a pointer to make it explicit. The Module owns the datalayout and it has to
  match the one attached to the TargetMachine for generating code.

* Comdats are now ortogonal to the linkage. LLVM will not create
  comdats for weak linkage globals and the frontends are responsible
  for explicitly adding them.

* On ELF we now support multiple sections with the same name and
  comdat. This allows for smaller object files since multiple
  sections can have a simple name (`.text`, `.rodata`, etc).

* LLVM now lazily loads metadata in some cases. Creating archives
  with IR files with debug info is now 25X faster.

* llvm-ar can create archives in the BSD format used by OS X.

* ... next change ...

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

 During this release ...


Changes to the PowerPC Target
-----------------------------

There are numerous improvements to the PowerPC target in this release:

* LLVM now supports the ISA 2.07B (POWER8) instruction set, including
  direct moves between general registers and vector registers, and
  built-in support for hardware transactional memory (HTM).  Some missing
  instructions from ISA 2.06 (POWER7) were also added.

* Code generation for the local-dynamic and global-dynamic thread-local
  storage models has been improved.

* Loops may be restructured to leverage pre-increment loads and stores.

* QPX - Hal, please say a few words.

* Loads from the TOC area are now correctly treated as invariant.

* PowerPC now has support for i128 and v1i128 types.  The types differ
  in how they are passed in registers for the ELFv2 ABI.

* Disassembly will now print shorter mnemonic aliases when available.

* Optional register name prefixes for VSX and QPX registers are now
  supported in the assembly parser.

* The back end now contains a pass to remove unnecessary vector swaps
  from POWER8 little-endian code generation.  Additional improvements
  are planned for release 3.8.

* The undefined-behavior sanitizer (UBSan) is now supported for PowerPC.

* Many new vector programming APIs have been added to altivec.h.
  Additional ones are planned for release 3.8.

* PowerPC now supports __builtin_call_with_static_chain.

* PowerPC now supports the revised -mrecip option that permits finer
  control over reciprocal estimates.

* Many bugs have been identified and fixed.


Changes to the OCaml bindings
-----------------------------

 During this release ...


External Open Source Projects Using LLVM 3.7
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.7.

* A project


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

