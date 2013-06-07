======================
LLVM 3.3 Release Notes
======================

.. contents::
    :local:

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

* MCJIT now supports exception handling. Support for it in the old jit will be
  removed in the 3.4 release.

* Command line options can now be grouped into categories which are shown in
  the output of ``-help``. See :ref:`grouping options into categories`.

* The appearance of command line options in ``-help`` that are inherited by
  linking with libraries that use the LLVM Command line support library can now
  be modified at runtime. See :ref:`cl::getRegisteredOptions`.

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

Removed support for legacy hexagonv2 and hexagonv3 processor architectures which
are no longer in use. Currently supported architectures are hexagonv4 and
hexagonv5.

Mips target
--------------

New features and improvements:

- Clang driver
 - Support for Sourcery CodeBench Mips toolchain directories tree.
 - Support for new command line options including:
  - -mxgot/-mno-xgot
  - -EL / -EB
  - -mmicromips / -mno-micromips
  - -msingle-float / -mdouble-float
  - -mabi=32 (o32 abi) and -mabi=64 (n64 abi)
 - Previously, options such as -mips16, -mmicromips, -mdsp and -mdspr2 were
   not passed to the assembler. This issue has been fixed.

- A number of changes have been made to improve the quality of DSP-ASE code
  generation.
 - Multiply and multiply-accumulate instructions can now use all four
   accumulators.
 - Instruction selection patterns have been added so that DSP instructions
   are emitted without having to use builtins.

- Delay slot filler pass can now search successor blocks for instructions to
  fill delay slots (use option -disable-mips-df-succbb-search=false).

PowerPC Target
--------------

New features and improvements:

- PowerPC now supports an assembly parser.
- Support added for thread-local storage.  64-bit ELF subtarget only.
- Support added for medium and large code model (-mcmodel=medium,large).
  Medium code model is now the default.  64-bit ELF subtarget only.
- Improved register allocation (fewer reserved registers).
- 64-bit atomic load and store are now supported.
- Improved code generation for unaligned memory accesses of scalar types.
- Improved performance of floating-point divide and square root
  with -ffast-math.
- Support for predicated returns.
- Improved code generation for comparisons.
- Support added for inline setjmp and longjmp.
- Support added for many instructions introduced in PowerISA 2.04, 2.05,
  and 2.06.
- Improved spill code for vector registers.
- Support added for -mno-altivec.
- ABI compatibility fixes for complex parameters, 128-bit integer parameters,
  and varargs functions.  64-bit ELF subtarget only.

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
default but can be enabled using the clang flag ``-fslp-vectorize``. The
BB-vectorizer can also be enabled using the command line flag
``-fslp-vectorize-aggressive``.

R600 Backend
------------

The R600 backend was added in this release, it supports AMD GPUs (HD2XXX -
HD7XXX).  This backend is used in AMD's Open Source graphics / compute drivers
which are developed as part of the `Mesa3D <http://www.mesa3d.org>`_ project.

SystemZ/s390x Backend
---------------------

LLVM and clang now support IBM's z/Architecture.  At present this support
is restricted to GNU/Linux (GNU triplet s390x-linux-gnu) and requires
z10 or greater.


Sub-project Status Update
=========================

In addition to the core LLVM 3.3 distribution of production-quality compiler
infrastructure, the LLVM project includes sub-projects that use the LLVM core
and share the same distribution license.  This section provides updates on these
sub-projects.


DragonEgg: GCC front-ends, LLVM back-end
----------------------------------------

`DragonEgg <http://dragonegg.llvm.org/>`_ is a
`GCC plugin <http://gcc.gnu.org/wiki/plugins>`_ that replaces GCC's optimizers
and code generators with LLVM's.  It works with gcc-4.5, 4.6, 4.7 and 4.8, can
target the x86-32/x86-64 and ARM processor families, and has been successfully
used on the Darwin, FreeBSD, KFreeBSD, Linux and OpenBSD platforms.  It fully
supports Ada, C, C++ and Fortran.  It has partial support for Go, Java, Obj-C
and Obj-C++.  Note that gcc-4.6 is the best supported version, and that Ada in
particular doesn't work well with gcc-4.7 and newer.

The `3.3 release <http://llvm.org/apt/>`_ has the following notable changes.

- supports gcc-4.8 (requires gcc-4.8.1 or newer)
- object files can be written directly using LLVM's integrated assembler
- produces saner debug info
- bitfields can now contain arbitrary scalar types (useful for Ada)


LLDB: Low Level Debugger
------------------------

`LLDB <http://lldb.llvm.org/>`_ is a ground-up implementation of a command-line
debugger, as well as a debugger API that can be used from scripts and other
applications. LLDB uses the following components of the LLVM core distribution
to support the latest language features and target support:

- the Clang parser for high-quality parsing of C, C++ and Objective C
- the LLVM disassembler
- the LLVM JIT compiler (MCJIT) for expression evaluation

The `3.3 release <http://llvm.org/apt/>`_ has the following notable changes.

Linux Features:

- Support for watchpoints
- vim integration for lldb commands and program status using a `vim plug-in
  <http://llvm.org/svn/llvm-project/lldb/trunk/utils/vim-lldb/README>`_
- Improved register support including vector registers
- Builds with cmake/ninja/auto-tools/clang 3.3/gcc 4.6

Linux Improvements:

- Debugging multi-threaded programs
- Debugging i386 programs
- Process list, attach and fork
- Expression evaluation


External Open Source Projects Using LLVM 3.3
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for a
lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.3.


Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL implementation,
another major goal of `pocl <http://pocl.sourceforge.net/>`_ is improving
performance portability of OpenCL programs with compiler optimizations, reducing
the need for target-dependent manual optimizations. An important part of pocl is
a set of LLVM passes used to statically parallelize multiple work-items with the
kernel compiler, even in the presence of work-group barriers. This enables
static parallelization of the fine-grained static concurrency in the work groups
in multiple ways.

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing new processors based
on the Transport triggered architecture (TTA).  The toolset provides a complete
co-design flow from C/C++ programs down to synthesizable VHDL/Verilog and
parallel program binaries.  Processor customization points include the register
files, function units, supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent
optimizations and also for parts of code generation. It generates new LLVM-based
code generators "on the fly" for the designed TTA processors and loads them in
to the compiler backend as runtime libraries to avoid per-target recompilation
of larger parts of the compiler chain.

Just-in-time Adaptive Decoder Engine (Jade)
-------------------------------------------

`Jade <https://github.com/orcc/jade>`_ (Just-in-time Adaptive Decoder Engine) is
a generic video decoder engine using LLVM for just-in-time compilation of video
decoder configurations. Those configurations are designed by MPEG Reconfigurable
Video Coding (RVC) committee. MPEG RVC standard is built on a stream-based
dataflow representation of decoders. It is composed of a standard library of
coding tools written in RVC-CAL language and a dataflow configuration --- block
diagram --- of a decoder.

Jade project is hosted as part of the Open RVC-CAL Compiler (`Orcc
<http://orcc.sf.net>`_) and requires it to translate the RVC-CAL standard
library of video coding tools into an LLVM assembly code.

LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X and Windows and also Linux/PPC64. Ports to
other architectures like ARM are underway.


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

