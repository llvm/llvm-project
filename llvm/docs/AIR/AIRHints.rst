==============================
Upstream AIR as an AGX backend
==============================

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

Portable AIR, and the one hint
==============================

The AIR is meant to be as portable as possible: nothing in it should need an
AGX (or a particular AGX generation) to make sense, so that the AIR side does
not grow with every new GPU (M3, M4 ...). The operations AIR does not have are
written as plain AIR that computes them, and the AGX backend recovers what
it can from that AIR on its own (``recoverAGXOperations`` in
``agx-lower-air``). A *hint* (below) is kept only where AGX has an operation
worth much more than its AIR form and cannot recover it from that form.

* Double precision arithmetic, compares, conversions, ``sqrt``, ``fma``,
  min / max, rounding: ``soft_f64.c`` (the AIR target's ``AIRSoftF64.inc``),
  doubles as ``i64``. AGX compiles the library (faster than its own software f64 on the
  PTX tests, -0.25% geomean; ``ddot`` +11% cycles).
* Constrained floating point (dynamic rounding, ``.rz``, ``.ftz``): to
  nearest, the AIR operation; directed, ``soft_f64.c``'s ``_r`` functions;
  flushes as selects. Folded at each use, where the mode is a constant almost
  everywhere; AGX drops ``.ftz``'s explicit flushes where its f32 ALU flushes
  anyway.
* ``llvm.agx.get.fpmode`` / ``set.fpmode``: the tracked mode (a constant) /
  nothing.
* ``llvm.minimum`` / ``maximum`` (and ``*num``): AIR's (``fmin`` / ``fmax``
  with NaN checks; through float for bfloat). AGX compiles that idiom as its
  minimum / maximum, in the narrow type.
* ``llvm.agx.rcp``, ``rsqrt``, ``rsqrt.special``: ``1 / x``,
  ``1 / sqrt(x)`` (afn), 0 and infinity selected. AGX compiles afn ``1 / x``
  and ``1 / sqrt(x)`` as rcp and rsqrt.
* ``llvm.agx.exp2``, ``log2``: ``llvm.exp2`` / ``log2`` (afn), which AGX
  compiles as its approximations already.
* ``llvm.agx.sin``, ``cos`` (argument in quadrants): ``sin(x * pi / 2)``
  (afn), which AGX compiles as the quadrant form.
* Scalar system values (SIMD lane, SIMD-group, thread index): AIR's (kernel
  parameters), read at the entry (+-0.01%).
* ``llvm.agx.core.id``: 0.
* Double ``atomicrmw``: a 64-bit ``cmpxchg`` loop with ``soft_f64.c`` (an
  LLVM operation AGX has; Metal does not).
* ``llvm.agx.simdgroup.matrix.fma``: **the hint** (below).

The emulations are inlined at each use by ``agx-raise-to-air`` and simplified
there (``agx-raise-hints`` selects the hints kept; ``-agx-raise-hints=all``
keeps every one, as the experiment first did). AIR has no floating point
environment, so the raise drops ``strictfp``.

Why the hints went (agx-bench, emulating each kind against the hint, before
the recovery above): system values +-0.01%; doubles -0.25% (the library is
faster); approximations +0.9% rw (nbody +36% tput), minimum / maximum up to
+72%, FP mode +0.9% co (``add_ftz`` +310%) -- all recovered by the AGX side
or by folding at the call. The matrix FMA as 24 shuffles and 16 FMAs costs
G14's tile GEMMs +15% cycles (geomean, worst +102%); G13, which expands it
anyway, is unchanged.

The hint
--------

A hint is an operation AIR does not have, written so that every AIR consumer
can still compile the module (``llvm/IR/AIR.h``): a call to an internal
``noinline`` function named ``__air_hint.<operation>``, whose attribute
``"air-hint"`` names the operation and whose body computes it in AIR (its
*emulation*). ``agx-lower-air`` replaces each call with the AGX operation; a
consumer that does not know the hint inlines the body (the Metal pipeline
inlines every function).

The only one kept is ``call:llvm.agx.simdgroup.matrix.fma.<type>|---|+lane``:
an 8x8 multiply-accumulate on per-thread fragments in AGX's layout
(AGXUsage.rst), whose emulation is 24 shuffles and 16 FMAs per thread and
reads the lane from its trailing ``+lane`` operand. AIR's own SIMD-group
matrices hold a whole 8x8 matrix as an opaque value; using them would need
the fragments to go through threadgroup memory, which an AGX lowering would
then have to remove again (future work).

Key formats (for ``-agx-raise-hints``): ``call:<intrinsic>|<doubles>[|<metadata
string>...][|+lane][|+mode][|+value]``, ``inst:<opcode>[:<predicate>]|<doubles>``,
``atomic:<operation>|<doubles>|<ordering>|<sync scope>|<align>``; ``<doubles>``
has a ``d`` for the result and each operand that is a double.

The floating point mode
-----------------------

PTX's rounding and flushing modifiers are AGX's floating point mode
(``llvm.agx.set.fpmode``: bits 0-1 the rounding of float, 2-3 of the other
formats, numbered like ``FLT_ROUNDS``; bit 4 flushes float denormals, bit 5
half's), which the constrained intrinsics with dynamic rounding read. AIR has
no such state, so ``agx-raise-to-air`` keeps the mode as a value: the mode AGX
starts a function with (``AGXLowerFPEnv``: rounding to nearest, the
function's denormal modes) and then what each ``set.fpmode`` writes, promoted
to SSA values, a constant wherever the mode is known; the emulations of the
constrained intrinsics are folded with it.

A directed rounding is computed in double precision: the operands of a half,
bfloat or float operation are doubles exactly, the ``_r`` functions of
``soft_f64.c`` round the double result in the direction, and rounding that
again to the format in the same direction rounds once (the values of the
format are doubles, so the directed rounding to the coarser format cannot
cross one). That holds for add, sub, mul, div, sqrt, fma and conversions from
integers. A dynamic rounding selects between the AIR operation (to nearest)
and the directed result.

Generic pointers and wide integers
----------------------------------

These are LLVM features AIR lacks rather than AGX operations, so they stay in
the AIR as LLVM writes them and each consumer legalizes them:

* Address space 4 (``air::GenericHintAS``): AGX's generic pointers.
  ``agx-lower-air`` maps it to AGX's generic address space and accepts casts
  to and from it; the portable passes leave atomics through such pointers as
  LLVM atomics. The Metal pipeline (``air-legalize-unsupported-ir``) makes
  them device pointers (wrong for a generic pointer to threadgroup memory).
* Integers of widths AIR does not have, up to 128 bits (128-bit atomics and
  accesses, the odd widths optimizations make). The Metal pipeline splits
  those of 65 to 128 bits into two i64 limbs: arithmetic, shifts, bitwise
  operations, comparisons, select, phi, loads and stores, bitcasts to
  128-bit vectors, ``abs``, ``ctlz``, ``cttz``, ``ctpop`` and min / max
  (checked against LLVM's own semantics with ``lli`` on 20000 random
  operands, i128 and i96). Atomics of those widths have no AIR form.

The Metal pipeline also turns 32-bit read-modify-writes that AIR has no
function for (``uinc_wrap`` / ``udec_wrap``: CUDA's ``atomicInc`` /
``atomicDec``, ``nand``, float min / max) into compare-exchange loops; the
portable form keeps them for AGX.

Testing
-------

``llc -agx-air-native-hints=false`` (``ZLUDA_AGX_LLVM_ARGS``) makes
``agx-lower-air`` run the emulation of the hints left (the matrix FMA), as a
consumer without the operation would; ``-agx-air-emulate-hints=<prefixes>``
does it for some.

Results (AGX Performance Simulator, 2026-10-28)
================================================

* Correctness, PTX-formalization (413 programs) through ZLUDA with
  ``ZLUDA_AGX_VIA_AIR=1``: 408 pass, the direct path's result (its own
  failures), with the matrix hint compiled natively and with it emulated.
* Speed, agx-bench against the direct path with the same ``llc`` flags (after
  agx-bench EXPERIMENTS.md L1 to L6): cycles / throughput geomean +0.48% /
  -0.10% on the CUDA suite (``ddot``: +18.7% cycles, -3.6% throughput; the
  portable soft-f64 issues fewer instructions but has a longer dependent
  chain), -5.41% / -3.45% on the PTX tests (``call_rnd`` -48%: directed
  float rounding by error-free transformations; device accesses that the
  direct path keeps generic now vectorize), -0.02% / +0.04% on agx_libs.
  Most of the AGX-side work in those rounds (store vectorization under
  register pressure, shaped vector accesses, shared divisor reciprocals,
  32-bit loop counters, bitfield extracts, 32-bit branch conditions) helps
  the direct path as much.
* The AIR: 2 of the 248 raised benchmark modules have a hint (the matrix
  FMA); the Metal pipeline writes a ``.metallib`` for 246 (the others need a
  128-bit atomic and a 64-bit compare-exchange). None was run on Metal here.

Not done
========

* Running the AIR on Metal (the Metal pipeline writes a ``.metallib`` for
  246 of the 248 raised benchmark modules; the others need a 128-bit atomic
  and a 64-bit compare-exchange, which AIR does not have).
* The matrix FMA in AIR's own SIMD-group matrices, with an AGX lowering that
  removes the threadgroup-memory round trip (the last hint).
* Bit-exact AGX behaviour is not a goal: the approximations, minimum /
  maximum of signed zeros and NaN payloads may differ between consumers.
* The mode across calls: each function starts from the default mode, as on
  AGX.
