======================
LLVM 3.3 Release Notes
======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 3.3 release.  You may
   prefer the `LLVM 3.2 Release Notes <http://llvm.org/releases/3.2/docs
   /ReleaseNotes.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.3.  Here we describe the status of LLVM, including major improvements
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

* The CellSPU port has been removed.  It can still be found in older versions.

* The IR-level extended linker APIs (for example, to link bitcode files out of
  archives) have been removed. Any existing clients of these features should
  move to using a linker with integrated LTO support.

* LLVM and Clang's documentation has been migrated to the `Sphinx
  <http://sphinx-doc.org/>`_ documentation generation system which uses
  easy-to-write reStructuredText. See `llvm/docs/README.txt` for more
  information.

* TargetTransformInfo (TTI) is a new interface that can be used by IR-level
  passes to obtain target-specific information, such as the costs of
  instructions. Only "Lowering" passes such as LSR and the vectorizer are
  allowed to use the TTI infrastructure.

* We've improved the X86 and ARM cost model.

* The Attributes classes have been completely rewritten and expanded. They now
  support not only enumerated attributes and alignments, but "string"
  attributes, which are useful for passing information to code generation. See
  :doc:`HowToUseAttributes` for more details.

* TableGen's syntax for instruction selection patterns has been simplified.
  Instead of specifying types indirectly with register classes, you should now
  specify types directly in the input patterns. See ``SparcInstrInfo.td`` for
  examples of the new syntax. The old syntax using register classes still
  works, but it will be removed in a future LLVM release.

* ... next change ...

.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

AArch64 target
--------------

We've added support for AArch64, ARM's 64-bit architecture. Development is still
in fairly early stages, but we expect successful compilation when:

- compiling standard compliant C99 and C++03 with Clang;
- using Linux as a target platform;
- where code + static data doesn't exceed 4GB in size (heap allocated data has
  no limitation).

Some additional functionality is also implemented, notably DWARF debugging,
GNU-style thread local storage and inline assembly.

Hexagon Target
--------------

- Removed support for legacy hexagonv2 and hexagonv3 processor
  architectures which are no longer in use. Currently supported
  architectures are hexagonv4 and hexagonv5.

Loop Vectorizer
---------------

We've continued the work on the loop vectorizer. The loop vectorizer now
has the following features:

- Loops with unknown trip counts.
- Runtime checks of pointers.
- Reductions, Inductions.
- Min/Max reductions of integers.
- If Conversion.
- Pointer induction variables.
- Reverse iterators.
- Vectorization of mixed types.
- Vectorization of function calls.
- Partial unrolling during vectorization.

The loop vectorizer is now enabled by default for -O3.

SLP Vectorizer
--------------

LLVM now has a new SLP vectorizer. The new SLP vectorizer is not enabled by
default but can be enabled using the clang flag -fslp-vectorize. The BB-vectorizer
can also be enabled using the command line flag -fslp-vectorize-aggressive.

R600 Backend
------------

The R600 backend was added in this release, it supports AMD GPUs
(HD2XXX - HD7XXX).  This backend is used in AMD's Open Source
graphics / compute drivers which are developed as part of the `Mesa3D
<http://www.mesa3d.org>`_ project.

SystemZ/s390x Backend
---------------------

LLVM and clang now support IBM's z/Architecture.  At present this support
is restricted to GNU/Linux (GNU triplet s390x-linux-gnu) and requires
z10 or greater.


External Open Source Projects Using LLVM 3.3
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.3.


Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL
implementation, another major goal of `pocl <http://pocl.sourceforge.net/>`_ 
is improving performance portability of OpenCL programs with
compiler optimizations, reducing the need for target-dependent manual
optimizations. An important part of pocl is a set of LLVM passes used to
statically parallelize multiple work-items with the kernel compiler, even in
the presence of work-group barriers. This enables static parallelization of
the fine-grained static concurrency in the work groups in multiple ways.

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing new 
processors based on the Transport triggered architecture (TTA). 
The toolset provides a complete co-design flow from C/C++
programs down to synthesizable VHDL/Verilog and parallel program binaries.
Processor customization points include the register files, function units,
supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent
optimizations and also for parts of code generation. It generates new
LLVM-based code generators "on the fly" for the designed TTA processors and
loads them in to the compiler backend as runtime libraries to avoid
per-target recompilation of larger parts of the compiler chain.


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

