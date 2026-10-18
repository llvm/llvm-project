=============================================
User Guide for the AGX Backend (Experimental)
=============================================

.. contents::
   :local:
   :depth: 2

Introduction
============

The AGX backend generates machine code for the GPUs of Apple silicon (the
"AGX" architecture of the M1 and later SoCs). It is an experimental target:
build it with ``-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=AGX``.

Apple does not document the instruction set. Everything in this backend comes
from public reverse engineering:

* `applegpu <https://github.com/dougallj/applegpu>`_, the reference for the
  G13 instruction encodings. The MC layer is tested against it: every
  encoding produced by the assembler and the code generator decodes to the
  same instruction in applegpu's disassembler.
* The Mesa ``asahi`` driver, the reference for the code that the hardware
  expects in practice (control flow on the nesting counter, scoreboarding,
  memory addressing). Only facts about the hardware were taken from it.
* Measurements of instruction latencies on G14 (M2 Max), used for the
  scheduling models.
* Reverse engineering notes on the G16/G17 ("Apple9") generation. These are
  incomplete, which is why code generation for those GPUs is not supported
  yet (see :ref:`agx-apple9`).

The backend is modelled on the AMDGPU backend: the same split between code
generation pseudo instructions and real, per encoding family instructions,
the same structurization of control flow and the same layout of the
sources.

Target triples
==============

Use ``agx`` as the architecture, for example ``agx-apple-macosx`` or simply
``-mtriple=agx``. The triple has no effect on code generation beyond selecting
the target. Object files are ELF (``EM_AGX``, ``0x4147``, a value that is not
registered); they contain the raw instructions of each kernel in ``.text``.

Processors
==========

Use ``-mcpu`` (or the ``target-cpu`` function attribute) to select the GPU.

.. table:: AGX processors
   :name: agx-processors-table

   =============== ========== ================= =====================================
   Processor       Generation Encoding family   Products
   =============== ========== ================= =====================================
   ``generic``     G13        AGX2              Any G13-G15 GPU
   ``agx-g13g``    G13        AGX2              M1
   ``agx-g13s``    G13        AGX2              M1 Pro (``multi-die-coherency``)
   ``agx-g13c``    G13        AGX2              M1 Max (``multi-die-coherency``)
   ``agx-g13d``    G13        AGX2              M1 Ultra (``multi-die-coherency``)
   ``agx-g14g``    G14        AGX2              M2
   ``agx-g14s``    G14        AGX2              M2 Pro
   ``agx-g14c``    G14        AGX2              M2 Max
   ``agx-g14d``    G14        AGX2              M2 Ultra
   ``agx-g15g``    G15        AGX2              M3
   ``agx-g15s``    G15        AGX2              M3 Pro
   ``agx-g15c``    G15        AGX2              M3 Max
   ``agx-g15d``    G15        AGX2              M3 Ultra
   ``agx-g16g``    G16        Apple9            M4
   ``agx-g16s``    G16        Apple9            M4 Pro
   ``agx-g16c``    G16        Apple9            M4 Max
   ``agx-g17p``    G17        Apple9            A18 Pro
   ``agx-g17g``    G17        Apple9            M5
   =============== ========== ================= =====================================

G15 support assumes the G14 encoding; it has not been validated on hardware.

Target features
===============

Each generation implies a set of features. Features can also be enabled or
disabled individually with ``-mattr`` (or the ``target-features`` attribute),
which allows code for one generation to be restricted to the capabilities of
another.

.. table:: AGX target features
   :name: agx-features-table

   ============================== ==============================================
   Feature                        Description
   ============================== ==============================================
   ``agx2-encoding``              G13/G14 instruction encoding.
   ``apple9-encoding``            G16/G17 instruction encoding.
   ``fp16-arith``                 Native half precision arithmetic.
   ``int64-add``                  64-bit ``iadd``/``imadd`` operands.
   ``simd-shuffle``               SIMD-group and quad shuffles.
   ``simd-reduce``                SIMD-group and quad reductions and prefixes.
   ``simd-ballot``                Ballots.
   ``simd-matrix``                8x8 SIMD-group matrix multiply-accumulate.
   ``quad-shuffle-index-merge``   ``simd_shuffle`` merges the low index bits of
                                  each quad (informational).
   ``multi-die-coherency``        Device atomics set the coherency bit (multi-die
                                  G13 parts).
   ``threadgroup-atomics``        Threadgroup memory atomics.
   ``device-atomics``             Device memory atomics.
   ``stack-access``               Per-thread stack loads and stores.
   ``dynamic-caching``            Registers and threadgroup memory are
                                  allocated dynamically (G15 and later).
   ``ray-tracing``                Hardware ray tracing (not used by code
                                  generation).
   ``mesh-shading``               Hardware mesh shading (not used by code
                                  generation).
   ``source-lifetime``            ALU sources carry register release flags
                                  (Apple9).
   ``max-gprs-128``, ``-64``      Limit the general purpose registers used.
   ============================== ==============================================

Address spaces
==============

.. table:: AGX address spaces
   :name: agx-address-spaces-table

   ============= ============ ============= =================================
   Address space Name         Pointer size  Memory
   ============= ============ ============= =================================
   0             generic      64            Device memory
   1             device       64            Device memory
   3             threadgroup  32            Threadgroup (shared) memory
   4             constant     64            Device memory, read only
   5             private      32            Per-thread stack
   ============= ============ ============= =================================

Generic, device and constant pointers are interchangeable: casts between them
are no-ops. There is no generic addressing of threadgroup or private memory.

Kernel ABI
==========

Kernels use the ``agx_kernel`` calling convention and return ``void``. There
are no calls: every other function is inlined, and the backend reports an
error for calls that remain. A module without ``agx_kernel`` functions
compiles each external function as an entry point with the kernel ABI.

Kernel arguments are passed in uniform registers. They are packed in order in
16-bit units (``u0l``, ``u0h``, ``u1l`` ...), each argument aligned to its
size and 64-bit arguments to 32 bits; arguments are split into their legal
parts first (``i8`` and ``i1`` take 16 bits, vectors one unit per element).
The 32-bit values ``threadgroups_per_grid.x``, ``.y`` and ``.z`` follow the
last argument, aligned to 32 bits; ``llvm.agx.threadgroups.per.grid.*``
read them.

Before each kernel, the assembly printer emits comments with the resources
the driver must provide::

  // uniform registers: 18 x 16-bit (threadgroups_per_grid at u6)
  // general purpose registers: 4 x 32-bit
  // threadgroup memory: 0 bytes
  // stack size: 0 bytes

Each kernel ends with ``stop`` followed by eight ``trap`` instructions, since
the instruction prefetcher reads past the end of the program.

Threadgroup variables are globals in address space 3 without an initializer.
They are allocated per kernel in threadgroup memory, at offsets in order of
first use.

Device and constant variables (globals in address spaces 0, 1 and 4) live in
the *global data buffer*. When a module defines any, each kernel receives
the 64-bit address of that buffer in the uniform registers following
``threadgroups_per_grid`` (reported in the resource comments as ``global
data buffer at uN_uN+1``). The variables are laid out in module order, each
aligned to its alignment, and the initial contents of the buffer, with that
layout, are the ``.agx.globals`` section of the object. Initializers that
refer to the addresses of other globals would need relocations and are not
supported.

Execution model
===============

A SIMD-group has 32 threads. There is no scalar unit and no scalar branch:
every branch is executed by the SIMD-group as a whole, under an execution
mask kept in the 16-bit nesting counter ``r0l`` of each thread (a thread is
active when its counter is zero). The code generator therefore structurizes
every function, uniform branches included, and annotates it with:

* ``llvm.agx.if``: ``if_icmp r0l, nueq, cond, 0, 1`` deactivates the threads
  where the condition is false, followed by ``jmp_exec_none`` to skip the
  region when no thread remains.
* ``llvm.agx.else``: ``else_fcmp r0l, eq, 0.0, 0.0, 1`` swaps the threads of
  the innermost level.
* ``llvm.agx.end.cf``: ``pop_exec r0l, 1`` reactivates the threads of the
  innermost construct. It is emitted at the start of the reconvergence
  block.
* ``llvm.agx.loop.begin``: ``push_exec r0l, 1`` in the loop preheader.
* ``llvm.agx.loop``: ``while_icmp r0l, ueq, break, 0, 1`` at the latch
  deactivates the threads leaving the loop, and ``jmp_exec_any`` branches back
  while any thread remains.

The nesting counter is zeroed at the start of each kernel that uses control
flow. Blocks are laid out in the reverse post-order of the structured graph,
so constructs fall through into each other as in Mesa's output.

Memory loads, stack loads and atomics complete asynchronously. A ``wait`` is
inserted before the first instruction that reads or overwrites a pending
result, before barriers and at the end of each block.

Registers
=========

``r0`` to ``r127`` are the 32-bit general purpose registers, with 16-bit
halves ``rNl`` and ``rNh``. ``r0`` is reserved for the nesting counter. 64-bit
ALU values use aligned register pairs; memory operations accept any register
tuple. ``u0`` to ``u255`` are the uniform registers. They are never allocated;
copies from them are folded into the ALU sources and memory bases that can
read them (64-bit ALU sources cannot be uniform).

Code generation
===============

Supported:

* 16, 32 and 64-bit integer arithmetic, including 64-bit multiplication,
  shifts, comparisons and bit counting. Division is expanded in IR, except
  by constants.
* Half and single precision arithmetic, with source modifiers and 8-bit
  floating point immediates folded. Division uses ``rcp`` and a multiply
  with a Newton-Raphson step (omitted with ``arcp``/``afn`` and for
  ``half``); ``sin``/``cos`` reduce to quadrants for ``sin_pt_1`` and
  ``sin_pt_2``; ``exp``, ``log`` and ``pow`` go through ``exp2``/``log2``.
  Conversions between 64-bit integers and ``float`` are expanded.
* Device memory loads and stores of 8 to 64 bits with folded base, scaled
  index and constant offsets; threadgroup and stack memory; memory
  intrinsics (inline or as loops).
* Device and threadgroup 32-bit atomics. Other atomic operations become
  compare-and-swap loops; atomic loads and stores become atomic operations.
* SIMD-group and quad shuffles, ballots, reductions and prefixes, derivatives,
  barriers and special registers through the ``llvm.agx.*`` intrinsics.

There is no floating point environment: constrained floating point
intrinsics are compiled as the ordinary operations, with round to nearest
even.

Not supported: ``double``, vector memory operations (vectors are
scalarized), textures and graphics shaders, calls and recursion.

.. _agx-apple9:

Apple9 (G16/G17)
================

The G16 and G17 GPUs use a different instruction encoding, of which too
little is known to generate code reliably. The processors are defined so that
features and scheduling can be selected, but instruction emission reports
``no AGX instruction encoding for ... (the Apple9 encoding is not yet
supported)``. Pseudo instructions carry an ``Apple9Real`` field for the
real instruction of that family, to be filled in as encodings are confirmed.

Intrinsics
==========

See ``llvm/include/llvm/IR/IntrinsicsAGX.td``. The special register
intrinsics (``llvm.agx.thread.position.in.grid.x`` and so on) read
``get_sr`` registers, ``llvm.agx.read.sr`` reads any special register by
number, and ``llvm.agx.simd.reduce``/``prefix`` and
``llvm.agx.quad.reduce``/``prefix`` take the operation as an immediate:

.. table:: SIMD-group reduction operations
   :name: agx-simd-ops-table

   ===== ========= ===== =========
   Value Operation Value Operation
   ===== ========= ===== =========
   0     and       16    iadd
   1     fadd      20    smin
   2     or        22    smax
   3     fmul      28    umin
   4     xor       30    umax
   5     fmin
   7     fmax
   ===== ========= ===== =========

MLIR
====

MLIR targets AGX the way it targets AMDGPU:

* The ``agxir`` dialect wraps the ``llvm.agx.*`` intrinsics one to one, like
  ``rocdl``. Kernels carry the ``agxir.kernel`` attribute and use the
  ``agx_kernelcc`` calling convention.
* The ``agx`` dialect provides higher level operations, like ``amdgpu``:
  ``agx.shuffle`` (any type, SIMD-group or quad scope), ``agx.reduce``,
  ``agx.ballot`` and ``agx.threadgroup_barrier``. ``-convert-agx-to-agxir``
  lowers them.
* ``-convert-gpu-to-agxir`` lowers the ``gpu`` dialect, and
  ``-agxir-attach-target`` attaches ``#agxir.target`` attributes.
  ``-gpu-module-to-binary`` then produces AGX assembly, ELF objects or
  bitcode.

.. code-block:: console

  $ mlir-opt kernel.mlir --pass-pipeline='builtin.module(agxir-attach-target{chip=agx-g14g},gpu.module(convert-gpu-to-agxir),gpu-module-to-binary{format=isa})'
