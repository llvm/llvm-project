========================
LLVM 9.0.0 Release Notes
========================

.. contents::
    :local:

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 9.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.


Known Issues
============

These are issues that couldn't be fixed before the release. See the bug reports
for the latest status.

* `PR40547 <https://llvm.org/pr40547>`_ Clang gets miscompiled by GCC 9.


Non-comprehensive list of changes in this release
=================================================

* Two new extension points, namely ``EP_FullLinkTimeOptimizationEarly`` and
  ``EP_FullLinkTimeOptimizationLast`` are available for plugins to specialize
  the legacy pass manager full LTO pipeline.

* ``llvm-objcopy/llvm-strip`` got support for COFF object files/executables,
  supporting the most common copying/stripping options.

* The CMake parameter ``CLANG_ANALYZER_ENABLE_Z3_SOLVER`` has been replaced by
  ``LLVM_ENABLE_Z3_SOLVER``.

* The RISCV target is no longer "experimental" (see
  `Changes to the RISCV Target`_ below for more details).

* The ORCv1 JIT API has been deprecated. Please see
  `Transitioning from ORCv1 to ORCv2 <ORCv2.html#transitioning-from-orcv1-to-orcv2>`_.

* Support for target-independent hardware loops in IR has been added, with
  PowerPC and Arm implementations.


Noteworthy optimizations
------------------------

* LLVM will now remove stores to constant memory (since this is a
  contradiction) under the assumption the code in question must be dead.  This
  has proven to be problematic for some C/C++ code bases which expect to be
  able to cast away 'const'.  This is (and has always been) undefined
  behavior, but up until now had not been actively utilized for optimization
  purposes in this exact way.  For more information, please see:
  `bug 42763 <https://bugs.llvm.org/show_bug.cgi?id=42763>`_ and
  `post commit discussion <http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20190422/646945.html>`_.

* The optimizer will now convert calls to ``memcmp`` into a calls to ``bcmp`` in
  some circumstances. Users who are building freestanding code (not depending on
  the platform's libc) without specifying ``-ffreestanding`` may need to either
  pass ``-fno-builtin-bcmp``, or provide a ``bcmp`` function.

* LLVM will now pattern match wide scalar values stored by a succession of
  narrow stores. For example, Clang will compile the following function that
  writes a 32-bit value in big-endian order in a portable manner:

  .. code-block:: c

      void write32be(unsigned char *dst, uint32_t x) {
        dst[0] = x >> 24;
        dst[1] = x >> 16;
        dst[2] = x >> 8;
        dst[3] = x >> 0;
      }

  into the x86_64 code below:

  .. code-block:: asm

   write32be:
           bswap   esi
           mov     dword ptr [rdi], esi
           ret

  (The corresponding read patterns have been matched since LLVM 5.)

* LLVM will now omit range checks for jump tables when lowering switches with
  unreachable default destination. For example, the switch dispatch in the C++
  code below

  .. code-block:: c

     int g(int);
     enum e { A, B, C, D, E };
     int f(e x, int y, int z) {
       switch(x) {
         case A: return g(y);
         case B: return g(z);
         case C: return g(y+z);
         case D: return g(x-z);
         case E: return g(x+z);
       }
     }

  will result in the following x86_64 machine code when compiled with Clang.
  This is because falling off the end of a non-void function is undefined
  behaviour in C++, and the end of the function therefore being treated as
  unreachable:

  .. code-block:: asm

   _Z1f1eii:
           mov     eax, edi
           jmp     qword ptr [8*rax + .LJTI0_0]


* LLVM can now sink similar instructions to a common successor block also when
  the instructions have no uses, such as calls to void functions. This allows
  code such as

  .. code-block:: c

   void g(int);
   enum e { A, B, C, D };
   void f(e x, int y, int z) {
     switch(x) {
       case A: g(6); break;
       case B: g(3); break;
       case C: g(9); break;
       case D: g(2); break;
     }
   }

  to be optimized to a single call to ``g``, with the argument loaded from a
  lookup table.


Changes to the LLVM IR
----------------------

* Added ``immarg`` parameter attribute. This indicates an intrinsic
  parameter is required to be a simple constant. This annotation must
  be accurate to avoid possible miscompiles.

* The 2-field form of global variables ``@llvm.global_ctors`` and
  ``@llvm.global_dtors`` has been deleted. The third field of their element
  type is now mandatory. Specify `i8* null` to migrate from the obsoleted
  2-field form.

* The ``byval`` attribute can now take a type parameter:
  ``byval(<ty>)``. If present it must be identical to the argument's
  pointee type. In the next release we intend to make this parameter
  mandatory in preparation for opaque pointer types.

* ``atomicrmw xchg`` now allows floating point types

* ``atomicrmw`` now supports ``fadd`` and ``fsub``

Changes to building LLVM
------------------------

* Building LLVM with Visual Studio now requires version 2017 or later.


Changes to the AArch64 Backend
------------------------------

* Assembly-level support was added for: Scalable Vector Extension 2 (SVE2) and
  Memory Tagging Extensions (MTE).

Changes to the ARM Backend
--------------------------

* Assembly-level support was added for the Armv8.1-M architecture, including
  the M-Profile Vector Extension (MVE).

* A pipeline model was added for Cortex-M4. This pipeline model is also used to
  tune for cores where this gives a benefit too: Cortex-M3, SC300, Cortex-M33
  and Cortex-M35P.

* Code generation support for M-profile low-overhead loops.


Changes to the MIPS Target
--------------------------

* Support for ``.cplocal`` assembler directive.

* Support for ``sge``, ``sgeu``, ``sgt``, ``sgtu`` pseudo instructions.

* Support for ``o`` inline asm constraint.

* Improved support of GlobalISel instruction selection framework.
  This feature is still in experimental state for MIPS targets though.

* Various code-gen improvements, related to improved and fixed instruction
  selection and encoding and floating-point registers allocation.

* Complete P5600 scheduling model.


Changes to the PowerPC Target
-----------------------------

* Improved handling of TOC pointer spills for indirect calls

* Improve precision of square root reciprocal estimate

* Enabled MachinePipeliner support for P9 with ``-ppc-enable-pipeliner``.

* MMX/SSE/SSE2 intrinsics headers have been ported to PowerPC using Altivec.

* Machine verification failures cleaned, EXPENSIVE_CHECKS will run
  MachineVerification by default now.

* PowerPC scheduling enhancements, with customized PPC specific scheduler
  strategy.

* Inner most loop now always align to 32 bytes.

* Enhancements of hardware loops interaction with LSR.

* New builtins added, eg: ``__builtin_setrnd``.

* Various codegen improvements for both scalar and vector code

* Various new exploitations and bug fixes, e.g: exploited P9 ``maddld``.


Changes to the SystemZ Target
-----------------------------

* Support for the arch13 architecture has been added.  When using the
  ``-march=arch13`` option, the compiler will generate code making use of
  new instructions introduced with the vector enhancement facility 2
  and the miscellaneous instruction extension facility 2.
  The ``-mtune=arch13`` option enables arch13 specific instruction
  scheduling and tuning without making use of new instructions.

* Builtins for the new vector instructions have been added and can be
  enabled using the ``-mzvector`` option.  Support for these builtins
  is indicated by the compiler predefining the ``__VEC__`` macro to
  the value ``10303``.

* The compiler now supports and automatically generates alignment hints
  on vector load and store instructions.

* Various code-gen improvements, in particular related to improved
  instruction selection and register allocation.

Changes to the X86 Target
-------------------------

* Fixed a bug in generating DWARF unwind information for 32 bit MinGW

Changes to the AMDGPU Target
----------------------------

* Function call support is now enabled by default

* Improved support for 96-bit loads and stores

* DPP combiner pass is now enabled by default

* Support for gfx10


Changes to the RISCV Target
---------------------------

The RISCV target is no longer "experimental"! It's now built by default,
rather than needing to be enabled with ``LLVM_EXPERIMENTAL_TARGETS_TO_BUILD``.

The backend has full codegen support for the RV32I and RV64I base RISC-V
instruction set variants, with the MAFDC standard extensions. We support the
hard and soft-float ABIs for these targets. Testing has been performed with
both Linux and bare-metal targets, including the compilation of a large corpus
of Linux applications (through buildroot).


Changes to LLDB
===============

* Backtraces are now color highlighting in the terminal.

* DWARF4 (debug_types) and DWARF5 (debug_info) type units are now supported.

* This release will be the last where ``lldb-mi`` is shipped as part of LLDB.
  The tool will still be available in a `downstream repository on GitHub
  <https://github.com/lldb-tools/lldb-mi>`_.

External Open Source Projects Using LLVM 9
==========================================

Mull - Mutation Testing tool for C and C++
------------------------------------------

`Mull <https://github.com/mull-project/mull>`_ is an LLVM-based tool for
mutation testing with a strong focus on C and C++ languages.

Portable Computing Language (pocl)
----------------------------------

In addition to producing an easily portable open source OpenCL
implementation, another major goal of `pocl <http://portablecl.org/>`_
is improving performance portability of OpenCL programs with
compiler optimizations, reducing the need for target-dependent manual
optimizations. An important part of pocl is a set of LLVM passes used to
statically parallelize multiple work-items with the kernel compiler, even in
the presence of work-group barriers. This enables static parallelization of
the fine-grained static concurrency in the work groups in multiple ways.

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://openasip.org/>`_ is an open source toolset for designing customized
processors based on the Transport Triggered Architecture (TTA).
The toolset provides a complete co-design flow from C/C++
programs down to synthesizable VHDL/Verilog and parallel program binaries.
Processor customization points include register files, function units,
supported operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++/OpenCL C language support, target independent
optimizations and also for parts of code generation. It generates new
LLVM-based code generators "on the fly" for the designed TTA processors and
loads them in to the compiler backend as runtime libraries to avoid
per-target recompilation of larger parts of the compiler chain.


Zig Programming Language
------------------------

`Zig <https://ziglang.org>`_  is a system programming language intended to be
an alternative to C. It provides high level features such as generics, compile
time function execution, and partial evaluation, while exposing low level LLVM
IR features such as aliases and intrinsics. Zig uses Clang to provide automatic
import of .h symbols, including inline functions and simple macros. Zig uses
LLD combined with lazily building compiler-rt to provide out-of-the-box
cross-compiling for all supported targets.


LDC - the LLVM-based D compiler
-------------------------------

`D <http://dlang.org>`_ is a language with C-like syntax and static typing. It
pragmatically combines efficiency, control, and modeling power, with safety and
programmer productivity. D supports powerful concepts like Compile-Time Function
Execution (CTFE) and Template Meta-Programming, provides an innovative approach
to concurrency and offers many classical paradigms.

`LDC <http://wiki.dlang.org/LDC>`_ uses the frontend from the reference compiler
combined with LLVM as backend to produce efficient native code. LDC targets
x86/x86_64 systems like Linux, OS X, FreeBSD and Windows and also Linux on ARM
and PowerPC (32/64 bit). Ports to other architectures are underway.


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
