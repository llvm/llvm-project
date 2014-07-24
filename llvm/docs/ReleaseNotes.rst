======================
LLVM 3.5 Release Notes
======================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 3.5 release.  You may
   prefer the `LLVM 3.4 Release Notes <http://llvm.org/releases/3.4/docs
   /ReleaseNotes.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.5.  Here we describe the status of LLVM, including major improvements
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

* All backends have been changed to use the MC asm printer and support for the
  non MC one has been removed.

* Clang can now successfully self-host itself on Linux/Sparc64 and on
  FreeBSD/Sparc64.

* LLVM now assumes the assembler supports ``.loc`` for generating debug line
  numbers. The old support for printing the debug line info directly was only
  used by ``llc`` and has been removed.

* All inline assembly is parsed by the integrated assembler when it is enabled.
  Previously this was only the case for object-file output. It is now the case
  for assembly output as well. The integrated assembler can be disabled with
  the ``-no-integrated-as`` option.

* llvm-ar now handles IR files like regular object files. In particular, a
  regular symbol table is created for symbols defined in IR files, including
  those in file scope inline assembly.

* LLVM now always uses cfi directives for producing most stack
  unwinding information.

* The prefix for loop vectorizer hint metadata has been changed from
  ``llvm.vectorizer`` to ``llvm.loop.vectorize``.  In addition,
  ``llvm.vectorizer.unroll`` metadata has been renamed
  ``llvm.loop.interleave.count``.

* Some backends previously implemented Atomic NAND(x,y) as ``x & ~y``. Now 
  all backends implement it as ``~(x & y)``, matching the semantics of GCC 4.4
  and later.

.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

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

Since release 3.3, a lot of new features have been included in the ARM
back-end but weren't production ready (ie. well tested) on release 3.4.
Just after the 3.4 release, we started heavily testing two major parts
of the back-end: the integrated assembler (IAS) and the ARM exception
handling (EHABI), and now they are enabled by default on LLVM/Clang.

The IAS received a lot of GNU extensions and directives, as well as some
specific pre-UAL instructions. Not all remaining directives will be
implemented, as we made judgement calls on the need versus the complexity,
and have chosen simplicity and future compatibility where hard decisions
had to be made. The major difference is, as stated above, the IAS validates
all inline ASM, not just for object emission, and that cause trouble with
some uses of inline ASM as pre-processor magic.

So, while the IAS is good enough to compile large projects (including most
of the Linux kernel), there are a few things that we can't (and probably
won't) do. For those cases, please use ``-fno-integrated-as`` in Clang.

Exception handling is another big change. After extensive testing and
changes to cooperate with Dwarf unwinding, EHABI is enabled by default.
The options ``-arm-enable-ehabi`` and ``-arm-enable-ehabi-descriptors``,
which were used to enable EHABI in the previous releases, are removed now.

This means all ARM code will emit EH unwind tables, or CFI unwinding (for
debug/profiling), or both. To avoid run-time inconsistencies, C code will
also emit EH tables (in case they interoperate with C++ code), as is the
case for other architectures (ex. x86_64).

Changes to the MIPS Target
--------------------------

There has been a large amount of improvements to the MIPS target which can be
broken down into subtarget, ABI, and Integrated Assembler changes.

Subtargets
^^^^^^^^^^

Added support for Release 6 of the MIPS32 and MIPS64 architecture (MIPS32r6
and MIPS64r6). Release 6 makes a number of significant changes to the MIPS32
and MIPS64 architectures. For example, FPU registers are always 64-bits wide,
FPU NaN values conform to IEEE 754 (2008), and the unaligned memory instructions
(such as lwl and lwr) have been replaced with a requirement for ordinary memory
operations to support unaligned operations. Full details of MIPS32 and MIPS64
Release 6 can be found on the `MIPS64 Architecture page at Imagination
Technologies <http://www.imgtec.com/mips/architectures/mips64.asp>`_.

This release also adds experimental support for MIPS-IV, cnMIPS, and Cavium
Octeon CPU's.

Support for the MIPS SIMD Architecture (MSA) has been improved to support MSA
on MIPS64.

Support for IEEE 754 (2008) NaN values has been added.

ABI and ABI extensions
^^^^^^^^^^^^^^^^^^^^^^

There has also been considerable ABI work since the 3.4 release. This release
adds support for the N32 ABI, the O32-FPXX ABI Extension, the O32-FP64 ABI
Extension, and the O32-FP64A ABI Extension.

The N32 ABI is an existing ABI that has now been implemented in LLVM. It is a
64-bit ABI that is similar to N64 but retains 32-bit pointers. N64 remains the
default 64-bit ABI in LLVM. This differs from GCC where N32 is the default
64-bit ABI.

The O32-FPXX ABI Extension is 100% compatible with the O32-ABI and the O32-FP64
ABI Extension and may be linked with either but may not be linked with both of
these simultaneously. It extends the O32 ABI to allow the same code to execute
without modification on processors with 32-bit FPU registers as well as 64-bit
FPU registers. The O32-FPXX ABI Extension is enabled by default for the O32 ABI
on mips*-img-linux-gnu and mips*-mti-linux-gnu triples and is selected with
-mfpxx. It is expected that future releases of LLVM will enable the FPXX
Extension for O32 on all triples.

The O32-FP64 ABI Extension is an extension to the O32 ABI to fully exploit FPU's
with 64-bit registers and is enabled with -mfp64. This replaces an undocumented
and unsupported O32 extension which was previously enabled with -mfp64. It is
100% compatible with the O32-FPXX ABI Extension.

The O32-FP64A ABI Extension is a restricted form of the O32-FP64 ABI Extension
which allows interlinking with unmodified binaries that use the base O32 ABI.

Integrated Assembler
^^^^^^^^^^^^^^^^^^^^

The MIPS Integrated Assembler has undergone a substantial overhaul including a
rewrite of the assembly parser. It's not ready for general use in this release
but adventurous users may wish to enable it using ``-fintegrated-as``.

In this release, the integrated assembler supports the majority of MIPS-I,
MIPS-II, MIPS-III, MIPS-IV, MIPS-V, MIPS32, MIPS32r2, MIPS32r6, MIPS64,
MIPS64r2, and MIPS64r6 as well as some of the Application Specific Extensions
such as MSA. It also supports several of the MIPS specific assembler directives
such as ``.set``, ``.module``, ``.cpload``, etc.

External Open Source Projects Using LLVM 3.5
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.5.


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

