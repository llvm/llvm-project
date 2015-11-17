======================
LLVM 3.7 Release Notes
======================

.. contents::
    :local:

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

Major changes in 3.7.1
======================

* 3.7.0 was released with an inadvertent change to the signature of the C
  API function: LLVMBuildLandingPad, which made the C API incompatible with
  prior releases.  This has been corrected in LLVM 3.7.1.

  As a result of this change, 3.7.0 is not ABI compatible with 3.7.1.

  +----------------------------------------------------------------------------+
  | History of the LLVMBuildLandingPad() function                              |
  +===========================+================================================+
  | 3.6.2 and prior releases  | LLVMBuildLandingPad(LLVMBuilderRef,            |
  |                           |                     LLVMTypeRef,               |
  |                           |                     LLVMValueRef,              |
  |                           |                     unsigned, const char*)     |
  +---------------------------+------------------------------------------------+
  | 3.7.0                     | LLVMBuildLandingPad(LLVMBuilderRef,            |
  |                           |                     LLVMTypeRef,               |
  |                           |                     unsigned, const char*)     |
  +---------------------------+------------------------------------------------+
  | 3.7.1 and future releases | LLVMBuildLandingPad(LLVMBuilderRef,            |
  |                           |                     LLVMTypeRef,               |
  |                           |                     LLVMValueRef,              |
  |                           |                     unsigned, const char*)     |
  +---------------------------+------------------------------------------------+


Non-comprehensive list of changes in 3.7.0
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

* The ``DataLayout`` is no longer optional. All the IR level optimizations expects
  it to be present and the API has been changed to use a reference instead of
  a pointer to make it explicit. The Module owns the datalayout and it has to
  match the one attached to the TargetMachine for generating code.

  In 3.6, a pass was inserted in the pipeline to make the ``DataLayout`` accessible:
    ``MyPassManager->add(new DataLayoutPass(MyTargetMachine->getDataLayout()));``
  In 3.7, you don't need a pass, you set the ``DataLayout`` on the ``Module``:
    ``MyModule->setDataLayout(MyTargetMachine->createDataLayout());``

  The LLVM C API ``LLVMGetTargetMachineData`` is deprecated to reflect the fact
  that it won't be available anymore from ``TargetMachine`` in 3.8.

* Comdats are now orthogonal to the linkage. LLVM will not create
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

  Support for BPF has been present in the kernel for some time, but starting
  from 3.18 has been extended with such features as: 64-bit registers, 8
  additional registers registers, conditional backwards jumps, call
  instruction, shift instructions, map (hash table, array, etc.), 1-8 byte
  load/store from stack, and more.

  Up until now, users of BPF had to write bytecode by hand, or use
  custom generators. This release adds a proper LLVM backend target for the BPF
  bytecode architecture.

  The BPF target is now available by default, and options exist in both Clang
  (-target bpf) or llc (-march=bpf) to pick eBPF as a backend.

* Switch-case lowering was rewritten to avoid generating unbalanced search trees
  (`PR22262 <http://llvm.org/pr22262>`_) and to exploit profile information
  when available. Some lowering strategies are now disabled when optimizations
  are turned off, to save compile time.

* The debug info IR class hierarchy now inherits from ``Metadata`` and has its
  own bitcode records and assembly syntax
  (`documented in LangRef <LangRef.html#specialized-metadata-nodes>`_).  The debug
  info verifier has been merged with the main verifier.

* LLVM IR and APIs are in a period of transition to aid in the removal of
  pointer types (the end goal being that pointers are typeless/opaque - void*,
  if you will). Some APIs and IR constructs have been modified to take
  explicit types that are currently checked to match the target type of their
  pre-existing pointer type operands. Further changes are still needed, but the
  more you can avoid using ``PointerType::getPointeeType``, the easier the
  migration will be.

* Argument-less ``TargetMachine::getSubtarget`` and
  ``TargetMachine::getSubtargetImpl`` have been removed from the tree. Updating
  out of tree ports is as simple as implementing a non-virtual version in the
  target, but implementing full ``Function`` based ``TargetSubtargetInfo``
  support is recommended.

* This is expected to be the last major release of LLVM that supports being
  run on Windows XP and Windows Vista.  For the next major release the minimum
  Windows version requirement will be Windows 7.

Changes to the MIPS Target
--------------------------

During this release the MIPS target has:

* Added support for MIPS32R3, MIPS32R5, MIPS32R3, MIPS32R5, and microMIPS32.

* Added support for dynamic stack realignment. This is of particular importance
  to MSA on 32-bit subtargets since vectors always exceed the stack alignment on
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


Changes to the JIT APIs
-----------------------

* Added a new C++ JIT API called On Request Compilation, or ORC.

  ORC is a new JIT API inspired by MCJIT but designed to be more testable, and
  easier to extend with new features. A key new feature already in tree is lazy,
  function-at-a-time compilation for X86. Also included is a reimplementation of
  MCJIT's API and behavior (OrcMCJITReplacement). MCJIT itself remains in tree,
  and continues to be the default JIT ExecutionEngine, though new users are
  encouraged to try ORC out for their projects. (A good place to start is the
  new ORC tutorials under llvm/examples/kaleidoscope/orc).

Sub-project Status Update
=========================

In addition to the core LLVM 3.7 distribution of production-quality compiler
infrastructure, the LLVM project includes sub-projects that use the LLVM core
and share the same distribution license. This section provides updates on these
sub-projects.

Polly - The Polyhedral Loop Optimizer in LLVM
---------------------------------------------

`Polly <http://polly.llvm.org>`_ is a polyhedral loop optimization
infrastructure that provides data-locality optimizations to LLVM-based
compilers. When compiled as part of clang or loaded as a module into clang,
it can perform loop optimizations such as tiling, loop fusion or outer-loop
vectorization. As a generic loop optimization infrastructure it allows
developers to get a per-loop-iteration model of a loop nest on which detailed
analysis and transformations can be performed.

Changes since the last release:

* isl imported into Polly distribution

  `isl <http://repo.or.cz/w/isl.git>`_, the math library Polly uses, has been
  imported into the source code repository of Polly and is now distributed as part
  of Polly. As this was the last external library dependency of Polly, Polly can
  now be compiled right after checking out the Polly source code without the need
  for any additional libraries to be pre-installed.

* Small integer optimization of isl

  The MIT licensed imath backend using in `isl <http://repo.or.cz/w/isl.git>`_ for
  arbitrary width integer computations has been optimized to use native integer
  operations for the common case where the operands of a computation fit into 32
  bit and to only fall back to large arbitrary precision integers for the
  remaining cases. This optimization has greatly improved the compile-time
  performance of Polly, both due to faster native operations also due to a
  reduction in malloc traffic and pointer indirections. As a result, computations
  that use arbitrary precision integers heavily have been speed up by almost 6x.
  As a result, the compile-time of Polly on the Polybench test kernels in the LNT
  suite has been reduced by 20% on average with compile time reductions between
  9-43%.

* Schedule Trees

  Polly now uses internally so-called > Schedule Trees < to model the loop
  structure it optimizes. Schedule trees are an easy to understand tree structure
  that describes a loop nest using integer constraint sets to keep track of
  execution constraints. It allows the developer to use per-tree-node operations
  to modify the loop tree. Programatic analysis that work on the schedule tree
  (e.g., as dependence analysis) also show a visible speedup as they can exploit
  the tree structure of the schedule and need to fall back to ILP based
  optimization problems less often. Section 6 of `Polyhedral AST generation is
  more than scanning polyhedra
  <http://www.grosser.es/#pub-polyhedral-AST-generation>`_ gives a detailed
  explanation of this schedule trees.

* Scalar and PHI node modeling - Polly as an analysis

  Polly now requires almost no preprocessing to analyse LLVM-IR, which makes it
  easier to use Polly as a pure analysis pass e.g. to provide more precise
  dependence information to non-polyhedral transformation passes. Originally,
  Polly required the input LLVM-IR to be preprocessed such that all scalar and
  PHI-node dependences are translated to in-memory operations. Since this release,
  Polly has full support for scalar and PHI node dependences and requires no
  scalar-to-memory translation for such kind of dependences.

* Modeling of modulo and non-affine conditions

  Polly can now supports modulo operations such as A[t%2][i][j] as they appear
  often in stencil computations and also allows data-dependent conditional
  branches as they result e.g. from ternary conditions ala A[i] > 255 ? 255 :
  A[i].

* Delinearization

  Polly now support the analysis of manually linearized multi-dimensional arrays
  as they result form macros such as
  "#define 2DARRAY(A,i,j) (A.data[(i) * A.size + (j)]". Similar constructs appear
  in old C code written before C99, C++ code such as boost::ublas, LLVM exported
  from Julia, Matlab generated code and many others. Our work titled
  `Optimistic Delinearization of Parametrically Sized Arrays
  <http://www.grosser.es/#pub-optimistic-delinerization>`_ gives details.

* Compile time improvements

  Pratik Bahtu worked on compile-time performance tuning of Polly. His work
  together with the support for schedule trees and the small integer optimization
  in isl notably reduced the compile time.

* Increased compute timeouts

  As Polly's compile time has been notabily improved, we were able to increase
  the compile time saveguards in Polly. As a result, the default configuration
  of Polly can now analyze larger loop nests without running into compile time
  restrictions.

* Export Debug Locations via JSCoP file

  Polly's JSCoP import/export format gained support for debug locations that show
  to the user the source code location of detected scops.

* Improved windows support

  The compilation of Polly on windows using cmake has been improved and several
  visual studio build issues have been addressed.

* Many bug fixes

libunwind
---------

The unwind implementation which use to reside in `libc++abi` has been moved into
a separate repository.  This implementation can still be used for `libc++abi` by
specifying `-DLIBCXXABI_USE_LLVM_UNWINDER=YES` and
`-DLIBCXXABI_LIBUNWIND_PATH=<path to libunwind source>` when configuring
`libc++abi`, which defaults to `true` when building on ARM.

The new repository can also be built standalone if just `libunwind` is desired.

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
