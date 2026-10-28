==========================================
Upstream AIR as an AGX backend, with hints
==========================================

.. contents::
   :local:

This page describes an experiment: compiling AGX kernels through upstream
(portable) AIR, so that one AIR module serves both Asahi Linux, where the AGX
backend compiles it, and macOS, where the AIR target's Metal pipeline packages
it for Metal. The direct path (an AGX module straight to the AGX backend) is
unchanged; this path is separate and opt-in.

The pipeline
============

::

  AGX IR (ZLUDA's PTX front end, optimized)
    -> agx-raise-to-air                 (opt -mtriple=agx -passes=agx-raise-to-air)
    -> the AIR target's portable passes (llc -mtriple=air -air-portable)
    =  upstream AIR, written as .air bitcode
    -> agx-lower-air, the AGX backend   (llc -mtriple=agx)       on Asahi
    -> the AIR target's Metal pipeline  (llc -mtriple=air -filetype=obj) for Metal

``agx-raise-to-air`` (``llvm/lib/Target/AGX/AGXRaiseToAIR.cpp``) is the
inverse of ``agx-lower-air``. ZLUDA selects the path with
``ZLUDA_AGX_VIA_AIR=1`` (``llvm_zluda``'s ``compile_agx``); ``zoc`` then also
writes ``NAME.air``.

What the raise writes as AIR
----------------------------

* Kernels: ``agx_kernel`` functions (or, in a module without them, the
  uncalled external ``void`` functions) become ``"air-kernel"`` functions with
  the C calling convention.
* Parameters: a ``byref`` parameter in AGX's constant address space is a
  constant buffer (address space 2) and keeps its ``byref`` type, the exact
  size of the buffer (AIR otherwise describes a buffer by the type its uses
  load). A by-value parameter becomes such a buffer too, marked
  ``"air-by-value"``, keeping its string attributes (``"agx-hidden"``).
* Address spaces: private (5) is AIR's thread (0), constant (4) AIR's
  constant (2); device and threadgroup keep their numbers.
* System values (``llvm.agx.thread.position.in.threadgroup.x`` and the
  others) are calls to ``air.thread_position_in_threadgroup.v3i32`` and the
  other AIR system values; SIMD-group shuffles, ballots, reductions and
  exclusive scans are ``air.simd_*`` / ``air.quad_*``; the threadgroup barrier
  is ``air.wg.barrier`` (its fences stay LLVM fences).

Hints
=====

A hint is an operation AIR does not have, written so that every AIR consumer
can still compile the module (``llvm/IR/AIR.h``).

Function hints
--------------

A call to an internal ``noinline`` function named ``__air_hint.<operation>``,
whose attribute ``"air-hint"`` names the operation and whose body computes it
in AIR (its *emulation*). ``agx-lower-air`` replaces each call with the AGX
operation; a consumer that does not know the hint inlines the body (the
Metal pipeline inlines every function). A hint whose operation has no AIR
emulation has an ``unreachable`` body and ``"air-hint-no-emulation"``: only a
consumer with the operation can run it.

The operation is named by a key:

* ``call:<intrinsic>|<doubles>[|<metadata string>...][|+lane]``: an LLVM
  intrinsic; constrained floating point intrinsics carry their rounding and
  exception metadata strings. ``+lane`` is a trailing operand only the
  emulation reads (the lane of the thread, a system value that only a kernel
  can read).
* ``inst:<opcode>[:<predicate>]|<doubles>``: a double precision instruction.
* ``atomic:<operation>|<doubles>|<ordering>|<sync scope>|<align>``: a double
  precision ``atomicrmw``.

``<doubles>`` has one character for the result and each operand, ``d`` where
it is a double (``-`` otherwise): AIR has no double, so doubles are ``i64``
bits in AIR and ``agx-lower-air`` bitcasts them back.

=================================== ======================================= ==================================
Operation                           AGX (``agx-lower-air``)                 AIR emulation
=================================== ======================================= ==================================
double precision arithmetic,        the double operation (AGXSoftF64)       arith-emulate-f64's ``soft_f64.c``
compares, conversions, ``sqrt``,                                            (``AGXAIRSoftF64.inc``), rounded to
``fma``, min/max, rounding                                                  nearest; canonical NaNs with
                                                                            ``canonical-nan``
constrained floating point          the constrained intrinsic               the plain operation (rounded to
(dynamic rounding, ``.rz`` ...)                                             nearest)
``llvm.agx.get.fpmode`` / ``set``   the FP mode registers                   0 / nothing
``llvm.minimum`` / ``maximum`` /    the intrinsic                           the intrinsic (AIR's fmin / fmax
``minimumnum`` / ``maximumnum``                                             with NaN checks)
``llvm.agx.rcp``, ``rsqrt``,        the hardware approximation              ``1 / x``, ``1 / sqrt(x)`` (afn)
``rsqrt.special``
``llvm.agx.exp2``, ``log2``         the hardware approximation              ``llvm.exp2`` / ``log2`` (afn)
``llvm.agx.sin``, ``cos``           the hardware approximation              ``sin(x * pi / 2)`` (afn)
(argument in quadrants)
``llvm.agx.core.id``                the core number                         0
``llvm.agx.simdgroup.matrix.fma``   the matrix unit or AGX's expansion      24 shuffles and 16 FMAs per thread,
                                                                            in the layout of AGXUsage.rst
double ``atomicrmw``                the atomic                              none (no 64-bit AIR atomics)
=================================== ======================================= ==================================

Address-space and type hints
----------------------------

* Address space 4 (``air::GenericHintAS``): AGX's generic pointers.
  ``agx-lower-air`` maps it to AGX's generic address space and accepts casts
  to and from it; the portable passes leave atomics through such pointers as
  LLVM atomics. A consumer without generic pointers must treat them as
  device pointers (wrong for a generic pointer to threadgroup memory).
* Integers of widths AIR does not have, up to 128 bits (128-bit atomics and
  accesses, the odd widths optimizations make). A consumer without them must
  legalize them.

Testing the emulations
----------------------

``llc -agx-air-native-hints=false`` (``ZLUDA_AGX_LLVM_ARGS``) makes
``agx-lower-air`` run every hint's emulation, as a consumer without the
operations would, so the AIR emulations run on AGX (or the simulator).

Results (AGX Performance Simulator, 2026-10-27)
================================================

* Correctness, PTX-formalization (413 programs) through ZLUDA with
  ``ZLUDA_AGX_VIA_AIR=1``: every program has the direct path's result (408
  pass; the failures are the direct path's own). With every hint emulated, 381
  pass; the others need rounding modes (``cvt-*``, ``rounding-diff-*``,
  ``testfloat-rm/rp/rz``), double precision atomics, or AGX's exact
  minimum/maximum and conversions.
* Speed, agx-bench (cycles of one SIMD-group, and throughput at the
  occupancy of each kernel), against the direct path with the same ``llc``
  flags: geomean -0.01% (CUDA suite), -0.00% (PTX tests), -0.01% (agx_libs);
  no kernel more than 2% slower in cycles.
* Metal: the Metal pipeline writes a ``.metallib`` for 241 of the 248 raised
  modules of the benchmark sets; the others use the integer or atomic hints
  above. None was run on Metal here.

Not done
========

* The Metal side's handling of the address-space and type hints (and of the
  ``byref`` attribute on constant buffers), and running the AIR on Metal.
* Emulations of rounding modes (the constrained intrinsics and the FP mode)
  and of double precision atomics.
