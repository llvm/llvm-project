======================
LLVM 3.4 Release Notes
======================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.4.  Here we describe the status of LLVM, including major improvements
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

* This is expected to be the last release of LLVM which compiles using a C++98
  toolchain. We expect to start using some C++11 features in LLVM and other
  sub-projects starting after this release. That said, we are committed to
  supporting a reasonable set of modern C++ toolchains as the host compiler on
  all of the platforms. This will at least include Visual Studio 2012 on
  Windows, and Clang 3.1 or GCC 4.7.x on Mac and Linux. The final set of
  compilers (and the C++11 features they support) is not set in stone, but we
  wanted users of LLVM to have a heads up that the next release will involve
  a substantial change in the host toolchain requirements.

* The regression tests now fail if any command in a pipe fails. To disable it in
  a directory, just add ``config.pipefail = False`` to its ``lit.local.cfg``.
  See :doc:`Lit <CommandGuide/lit>` for the details.

* Support for exception handling has been removed from the old JIT. Use MCJIT
  if you need EH support.

* The R600 backend is not marked experimental anymore and is built by default.

* APFloat::isNormal() was renamed to APFloat::isFiniteNonZero() and
  APFloat::isIEEENormal() was renamed to APFloat::isNormal(). This ensures that
  APFloat::isNormal() conforms to IEEE-754R-2008.

* The library call simplification pass has been removed.  Its functionality
  has been integrated into the instruction combiner and function attribute
  marking passes.

* Support for building using Visual Studio 2008 has been dropped. Use VS 2010
  or later instead. For more information, see the `Getting Started using Visual
  Studio <GettingStartedVS.html>`_ page.

* The Loop Vectorizer that was previously enabled for -O3 is now enabled for
  -Os and -O2.

* The new SLP Vectorizer is now enabled by default.

* llvm-ar now uses the new Object library and produces archives and
  symbol tables in the gnu format.

* FileCheck now allows specifing -check-prefix multiple times. This
  helps reduce duplicate check lines when using multiple RUN lines.

* The bitcast instruction no longer allows casting between pointers
   with different address spaces. To achieve this, use the new
   addrspacecast instruction.

* Different sized pointers for different address spaces should now
  generally work. This is primarily useful for GPU targets.

* ... next change ...

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Mips Target
-----------

Support for the MIPS SIMD Architecture (MSA) has been added. MSA is supported
through inline assembly, intrinsics with the prefix '__builtin_msa', and normal
code generation.

For more information on MSA (including documentation for the instruction set),
see the `MIPS SIMD page at Imagination Technologies
<http://imgtec.com/mips/mips-simd.asp>`_

External Open Source Projects Using LLVM 3.4
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.4.


LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X, FreeBSD and Windows and also Linux/PPC64.
Ports to other architectures like ARM and AArch64 are underway.


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

