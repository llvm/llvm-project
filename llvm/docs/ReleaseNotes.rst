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
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
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

* LLVM received a backend for the extended Berkely Packet Filter
  instruction set that can be dynamically loaded into the Linux kernel via the
  `bpf(2) <http://man7.org/linux/man-pages/man2/bpf.2.html>`_ syscall.

* Switch-case lowering was rewritten to avoid generating unbalanced search trees
  (`PR22262 <http://llvm.org/pr22262>`_) and to exploit profile information
  when available. Some lowering strategies are now disabled when optimizations
  are turned off, to save compile time.

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

During this release the MIPS target has:

* Added support for MIPS32R3, MIPS32R5, MIPS32R3, MIPS32R5, and microMIPS32.

* Added support for dynamic stack realignment. This of particular importance to
  MSA on 32-bit subtargets since vectors always exceed the stack alignment on
  the O32 ABI.

* Added support for compiler-rt including:

  * Support for the Address, and Undefined Behaviour Sanitizers for all MIPS
    subtargets.

  * Support for the Data Flow, and Memory Sanitizer for 64-bit subtargets.

  * Support for the Profiler for all MIPS subtargets.

* Added support for libcxx, and libcxxabi.

* Improved inline assembly support such that memory constraints may now make use
  of the appropriate address offsets available to the instructions. Also, added
  support for the ``ZC`` constraint.

* Added support for 128-bit integers on 64-bit subtargets and 16-bit floating
  point conversions on all subtargets.

* Added support for read-only ``.eh_frame`` sections by storing type information
  indirectly.

* Added support for MCJIT on all 64-bit subtargets as well as MIPS32R6.

* Added support for fast instruction selection on MIPS32 and MIPS32R2 with PIC.

* Various bug fixes. Including the following notable fixes:

  * Fixed 'jumpy' debug line info around calls where calculation of the address
    of the function would inappropriately change the line number.

  * Fixed missing ``__mips_isa_rev`` macro on the MIPS32R6 and MIPS32R6
    subtargets.

  * Fixed representation of NaN when targeting systems using traditional
    encodings. Traditionally, MIPS has used NaN encodings that were compatible
    with IEEE754-1985 but would later be found incompatible with IEEE754-2008.

  * Fixed multiple segfaults and assertions in the disassembler when
    disassembling instructions that have memory operands.

  * Fixed multiple cases of suboptimal code generation involving $zero.

  * Fixed code generation of 128-bit shifts on 64-bit subtargets.

  * Prevented the delay slot filler from filling call delay slots with
    instructions that modify or use $ra.

  * Fixed some remaining N32/N64 calling convention bugs when using small
    structures on big-endian subtargets.

  * Fixed missing sign-extensions that are required by the N32/N64 calling
    convention when generating calls to library functions with 32-bit
    parameters.

  * Corrected the ``int64_t`` typedef to be ``long`` for N64.

  * ``-mno-odd-spreg`` is now honoured for vector insertion/extraction
    operations when using -mmsa.

  * Fixed vector insertion and extraction for MSA on 64-bit subtargets.

  * Corrected the representation of member function pointers. This makes them
    usable on microMIPS subtargets.

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

* QPX - The vector instruction set used by the IBM Blue Gene/Q supercomputers
  is now supported.

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

Changes to the SystemZ Target
-----------------------------

* LLVM no longer attempts to automatically detect the current host CPU when
  invoked natively.

* Support for all thread-local storage models. (Previous releases would support
  only the local-exec TLS model.)

* The POPCNT instruction is now used on z196 and above.

* The RISBGN instruction is now used on zEC12 and above.

* Support for the transactional-execution facility on zEC12 and above.

* Support for the z13 processor and its vector facility.


Changes to the OCaml bindings
-----------------------------

 During this release ...


External Open Source Projects Using LLVM 3.7
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for
a lot of other language and tools projects. This section lists some of the
projects that have already been updated to work with LLVM 3.7.


LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X, FreeBSD and Windows and also Linux on
PowerPC (32/64 bit). Ports to other architectures like ARM, AArch64 and MIPS64
are underway.

Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL
implementation, another major goal of `pocl <http://portablecl.org/>`_
is improving performance portability of OpenCL programs with
compiler optimizations, reducing the need for target-dependent manual
optimizations. An important part of pocl is a set of LLVM passes used to
statically parallelize multiple work-items with the kernel compiler, even in
the presence of work-group barriers.


TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing customized
exposed datapath processors based on the Transport triggered
architecture (TTA).

The toolset provides a complete co-design flow from C/C++
programs down to synthesizable VHDL/Verilog and parallel program binaries.
Processor customization points include the register files, function units,
supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent
optimizations and also for parts of code generation. It generates
new LLVM-based code generators "on the fly" for the designed processors and
loads them in to the compiler backend as runtime libraries to avoid
per-target recompilation of larger parts of the compiler chain.

BPF Compiler Collection (BCC)
-----------------------------
`BCC <https://github.com/iovisor/bcc>`_ is a Python + C framework for tracing and
networking that is using Clang rewriter + 2nd pass of Clang + BPF backend to
generate eBPF and push it into the kernel.

LLVMSharp & ClangSharp
----------------------

`LLVMSharp <http://www.llvmsharp.org>`_ and
`ClangSharp <http://www.clangsharp.org>`_ are type-safe C# bindings for
Microsoft.NET and Mono that Platform Invoke into the native libraries.
ClangSharp is self-hosted and is used to generated LLVMSharp using the
LLVM-C API.

`LLVMSharp Kaleidoscope Tutorials <http://www.llvmsharp.org/Kaleidoscope/>`_
are instructive examples of writing a compiler in C#, with certain improvements
like using the visitor pattern to generate LLVM IR.

`ClangSharp PInvoke Generator <http://www.clangsharp.org/PInvoke/>`_ is the
self-hosting mechanism for LLVM/ClangSharp and is demonstrative of using
LibClang to generate Platform Invoke (PInvoke) signatures for C APIs.


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

