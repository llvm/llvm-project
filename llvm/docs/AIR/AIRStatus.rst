=============================
AIR Status and Known Blockers
=============================

.. contents::
   :local:

This page records what has and has not been verified for the AIR target and
the AIR to AGX path, and the blockers found while bringing them up. Each entry
says what is blocked, the evidence, and what would unblock it.

Verification so far
===================

* Output of this backend runs on a real Apple M2 Max through the AIR service
  (``/air``, a file-queue RPC that accepts AIR bitcode only and refuses to
  compile Metal source). ``verify_gpu.py --extended`` passes on a two-kernel
  ``.air`` produced with ``-air-output=air``.
* ``llvm/utils/air-gpu`` runs generated kernels on the M2 Max and checks every
  result against a reference computed in Python. All of these groups pass:

  ================= =========================================================
  Group             Coverage
  ================= =========================================================
  ``llvm-math-f32`` every LLVM float intrinsic with an AIR function
  ``llvm-math-afn`` the same with ``afn``, which selects ``air.fast_*``
  ``llvm-int``      ``smin smax umin umax abs ctpop ctlz cttz bitreverse``
  ``air-math-f32``  the AIR math functions called directly, precise and fast
  ``simd``          shuffles (index, xor, up, down, quad xor), sum, min, max,
                    prefix sums, and/or/xor, ballot, vote, for i32/u32/f32
  ``atomics``       ``atomicrmw`` (all operations, both orderings), ``fadd``,
                    ``cmpxchg``, atomic load/store, fences
  ``threadgroup``   threadgroup memory with explicit 2- and 3-operand
                    barriers, inserted barriers, threadgroup atomics, all
                    with two SIMD-groups per threadgroup
  ``sysvals``       all eleven compute system values
  ``types``         half, ``<4 x float>``, i8/i16/i64, 64-bit division,
                    conversions
  ``memory``        struct GEPs through a pointer select, nested aggregates
                    through a pointer phi, private arrays, divergent loops,
                    constant tables and buffers, ``memcpy``, scalar arguments
  ``matrix``        8x8 SIMD-group matrices: f32, f16, accumulate, transposed,
                    from threadgroup memory (results exact)
  ``misc``          device-coherent loads and stores, ``air.mul_hi``
  ``extra``         builtin ids, ``convert``, ``dot``, ``pack``, ``sincos``,
                    fill shuffles, quad ballot, vote all, SIMD-group barrier
  ================= =========================================================

  In all, 230 cases pass. One probe records a function AIR does not have
  (``air.copysign``), and one case needs a newer OS than the device runs
  (below).

* No AIR in any test or fixture was produced by Apple's Metal compiler. The
  ``.metallib`` and ``.air`` files in ``Metal.jl/test/metallib`` are built
  from ``.metal`` sources with Apple's toolchain and are deliberately unused.

Defects found on hardware and fixed
===================================

Each of these passed the FileCheck tests of the original branch.

* Kernels reading a system value lost their ``!air.kernel`` entry, so
  ``-filetype=obj`` silently wrote bare bitcode instead of a ``.metallib``.
* Thirteen LLVM math intrinsics were not lowered (``tan``, ``asin``,
  ``acos``, ``atan``, ``atan2``, ``sinh``, ``cosh``, ``tanh``, ``exp10``,
  ``log10``, ``round``, ...): Metal reports undefined symbols, and for
  ``round`` and ``log10`` its compiler service crashes. Precise intrinsics
  were lowered to relaxed functions (``llvm.sqrt`` to ``air.fast_sqrt``,
  1 ulp off).
* ``atomicrmw umax``/``umin``/``fadd`` were given names that crash Metal's
  compiler service; every atomic was emitted relaxed and with threadgroup
  scope whatever its ordering and memory; ``cmpxchg`` and ``fence`` were
  not lowered at all (the writer stopped).
* The writer emitted an out-of-range type index for a call whose function
  type differs from its callee's, giving bitcode no reader accepts.
* Barriers were not inserted for threadgroup read after write, and were
  inserted in the two-operand form in modules using the three-operand one.
* ``air.threads_per_grid`` and ``air.thread_index_in_threadgroup`` were not
  lowered.
* A constant-address-space buffer argument was taken for a Triton scalar
  passed by reference: it was packed away and its uses replaced with
  ``undef``, so the kernel read zeros.
* Plain LLVM IR the Metal pipeline passed through and Metal gets wrong (groups
  ``llvm-float-ieee``, ``llvm-int-ops``, ``llvm-conv``, ``llvm-memory-ir``),
  now expanded by ``air-legalize-unsupported-ir`` and ``air-nan-min-max``
  in the Metal pipeline only (the portable AIR for AGX is unchanged):
  ``frem`` (inexact, NaN for ``x frem inf``) becomes ``air.fmod.f32``;
  ``llvm.round`` gets the operand's sign back (``air.round(-0)`` is +0);
  ``llvm.minimum``/``maximum`` order signed zeros; ``llvm.roundeven`` and
  ``llvm.nearbyint`` (compiler crash) become ``air.rint``;
  ``llvm.sadd``/``ssub.with.overflow`` (wrong overflow bit) and
  ``llvm.fptosi``/``fptoui.sat`` (compiler crash) are expanded;
  ``llvm.ldexp``/``llvm.frexp`` (undefined) become ``air.ldexp(x, i32)`` and
  ``air.frexp(x, thread i32 *)``; ``llvm.is.fpclass`` (undefined) becomes
  integer tests. A private ``alloca`` accessed with another type (an i8 GEP,
  an i32 store into an array, ``llvm.memset``) made the library rejected
  (``Unexpected bitcode file!``): the writer typed the alloca's pointer by its
  uses instead of its allocated type.

Blockers
========

The AIR service reports no diagnostics for rejected AIR
---------------------------------------------------------

*What:* AIR that the service's host tool cannot handle is rejected with
``{"ok": false, "error": "Tool failed"}`` and nothing else. Only errors the
tool prints as ``LLVM ERROR: ...`` come back verbatim (for example
``Module is missing AIR version number``). The request schema is strict:
unknown fields such as ``verbose`` or ``debug`` are rejected.

*Impact:* finding why a module is rejected is bisection by hand. Rejections
seen so far were all from malformed kernel metadata (next entry), found by
bisecting with minimal kernels.

*Unblock:* have the service return the host tool's exit status and stderr.

Pre-baked minimal ``!air.kernel`` entries produce AIR the device rejects
-------------------------------------------------------------------------

*What:* several tests in ``llvm/test/CodeGen/AIR`` (inherited from the
original branch) mark kernels with ``!air.kernel = !{!{ptr @f}}``. When an
``!air.kernel`` node is present, ``AIRSystemValues`` keeps it as is instead
of synthesizing the full ``!{ptr @f, !{outputs}, !{arguments}}`` form, so the
output has no argument descriptions. Those tests only check the container
magic, so they pass, but the service rejects such a module (``Tool failed``)
even for an empty kernel. Without a pre-baked node the pass synthesizes the
full form and the device accepts it.

*Unblock:* either complete bare entries in ``AIRSystemValues`` or reject
them with a diagnostic, and move the tests to unmarked kernels.

Apple's ``metallib`` tool cannot be run here
--------------------------------------------

*What:* the ``.air`` emitted by ``-air-output=air`` is the input format of
Apple's ``metallib`` linker. There is no macOS host in this environment and
the service does not expose the tool directly, so ``.air`` to ``.metallib``
linking with ``xcrun metallib`` is not verified here. (The service itself
accepts the ``.air`` and runs it, which exercises an equivalent load path.)

*Unblock:* run ``xcrun metallib k.air -o k.metallib`` on a Mac.

The AGX Performance Simulator needed a Rust toolchain (resolved)
----------------------------------------------------------------

*What:* the simulator in ``/scratch/AGX-Performance-Simulator`` is a Rust
crate (``agx_run``, ``agx_cost``) and the container had no ``cargo`` or
``rustc``. A standalone Rust 1.89.0 installed without root now builds both.

The GitHub key here is scoped to one repository
-----------------------------------------------

*What:* the SSH key answers ``Hi Jianwen-Ding/llvm-project!``: a deploy key
for that repository alone. It can push this fork but cannot read any other
repository, including the simulator's remote.

AIR features with no AGX lowering
---------------------------------

*What:* ``agx-lower-air`` (see "AIR to AGX" below) reports an error, rather
than miscompiling, for the AIR it has no AGX equivalent for yet: SIMD-group
matrices (``air.simdgroup_matrix_8x8_*``), SIMD-group async copies, tensors
(``air.*_private_tensor``), argument buffers (``air.indirect_buffer``),
function groups and more than one threadgroup buffer. The AGX backend has no
lowering of ``llvm.tan``, ``asin``, ``acos``, ``atan``, ``atan2``, ``sinh``,
``cosh`` and ``tanh``, nor of AIR's ``sinpi``, ``cospi``, ``tanpi``,
``asinh``, ``acosh`` and ``atanh``.

Device and service limits
=========================

The AIR service hides how Metal fails
-------------------------------------

Besides the plain ``Tool failed`` above, the service reports
``XPC_ERROR_CONNECTION_INTERRUPTED`` both for transient failures and when
Metal's compiler service crashes on the library it was given; only retrying
tells them apart. Several backend defects showed up only as such crashes.

``llvm.sqrt`` is not correctly rounded through Metal as it is
-------------------------------------------------------------

Passed to Metal unchanged, ``llvm.sqrt.f32`` is 1 ulp off; ``air.sqrt.f32``
is correctly rounded, which is why the intrinsic is now lowered to it.

Float subnormals are flushed to zero
------------------------------------

The M2 Max flushes f32 subnormal operands and results to zero, so results
that depend on them (``llvm-float-ieee``'s ``denorm_*`` cases, a minimum of
+0 and a subnormal) differ from LLVM's IEEE default. Not fixable in the
backend.

Weak compare-and-exchange
-------------------------

``air.atomic.*.cmpxchg.weak`` returns the value it found, which equals the
expected value both on success and on a spurious failure. The backend lowers
LLVM's strong ``cmpxchg`` to a single call and treats a match as success;
on the M2 Max no spurious failure was seen. If Apple GPUs can fail it
spuriously, a strong ``cmpxchg`` can report a success that did not happen.

The device's OS accepts AIR 2.8, not 2.9
----------------------------------------

The M2 Max behind the service rejects AIR 2.9 modules (``air version set to
2.9.0 ..., but expecting 2.8``): its OS has MSL 4.0. The compare-and-exchange
with a memory-flags operand exists only from AIR 2.9 (MSL 4.1), so modules
using it, for which the backend raises ``air.version`` (with a warning), are
unverified here. Apple does not document which OS versions bring MSL 4.1, so
the target triple cannot express it yet.

What the service cannot run
---------------------------

``execute`` dispatches compute kernels only, so vertex, fragment, mesh and
tessellation functions cannot be run, and function constants cannot be set.

Since 2026-10-27 it also binds textures (1D, 2D, 3D; 15 formats from
``r8unorm`` to ``rgba32float``) and samplers (nearest, linear), with
``/air/client/verify_textures.py`` as acceptance test. This backend has not
yet been made to emit texture code (texture handle types,
``air.sample_texture_*``/``read``/``write`` calls, ``air.texture`` and
``air.sampler`` argument metadata); that, and a texture group in
``llvm/utils/air-gpu``, is the next step for compute textures.

AGX Performance Simulator gaps
==============================

Sub-32-bit kernel arguments cannot be set
-----------------------------------------

The AGX ABI packs kernel arguments in 16-bit halves of the uniform registers,
but ``agx_run --uniform I=V`` sets whole 32-bit registers, and nothing says
how to set one half. ``llvm/utils/agx-checked/run_checked.py`` writes
``uniforms[half // 2]``, which overwrites the other half: a 16- or 8-bit
argument at an odd half, or the argument sharing its register, gets a wrong
value.

No real AGX hardware
--------------------

AGX code is only run on the simulator here: there is no Apple silicon
machine running Linux in this environment, so a passing differential test
(same kernel through AIR on the M2 Max, through AGX on the simulator) shows
agreement with the simulator's model, not with the hardware.

Apple-compiled code within reach
================================

Closed-source compiled AIR is present in the workspace and could enter a test
by accident: ``Metal.jl/test/metallib`` holds ``.metallib`` and ``.air``
files built from ``.metal`` sources by Apple's compiler (its ``Makefile``
shows how), and ``floor/etc/example/nbody.metallib`` is another. None is
used. ``llvm/utils/air-gpu`` generates every kernel itself, and its README
says not to add such files. The service accepts any AIR it is sent, so it
would not catch one.

Not done yet
============

MLIR AIR target
---------------

The MLIR ``air`` dialect and ``Target/LLVM/AIR`` were cherry-picked onto
LLVM 24, with their registration resolved next to AGX's, but ``mlir-opt`` has
not been built here (the build was interrupted, with no errors up to then) and
``module-to-binary-air.mlir`` has not been run.

AIR to AGX
----------

``agx-lower-air`` (``llvm/lib/Target/AGX/AGXLowerAIR.cpp``, documented in
``AGXUsage.rst``, "AIR input") compiles the AIR that ``addAIRPortablePasses``
produces, in process: the module after ``air-system-values``, before the
passes that adapt it to the Metal driver and before the bitcode writer. It
does not read ``.air`` files. It accepts only AIR's types, rebuilds each
kernel as an ``agx_kernel`` with its buffers in location order (constant
buffers by reference, so in uniform registers), turns system values into
special register reads, remaps the address spaces, lowers the math functions
by the inverse of ``AIRIntrinsicMappings.td`` (generated in the AGX target
from the same file), atomics and fences to ``atomicrmw``, ``cmpxchg`` and
``fence``, and barriers to ``llvm.agx.threadgroup_barrier`` between
threadgroup-scope fences.

``llvm/utils/air-gpu/air_gpu_suite.py --agx-run`` runs the suite's kernels
through the portable passes and ``agx-lower-air`` on the AGX Performance
Simulator (``agx_run``, G14G), against the same references as on the M2 Max.
On 2026-10-27: 172 pass (the matrix group exactly), 42 do not compile (the
transcendental functions above), the probe for ``air.copysign``
is rejected as expected, and one case is skipped: it has no barrier and
relies on ``air-tg-barrier-insert``, which only the Metal pipeline runs. 16
compute the right function less accurately than MSL requires of its precise
functions: ``sqrt`` (1 ulp), ``sin`` (24, 132 for ``<4 x float>``), ``cos``
(207), ``log2`` (10), ``log10`` (39), ``exp10`` (7), ``pow`` (20) and
``sincos`` (23), the accuracy of the AGX backend's lowering of these
intrinsics. The results are the simulator's model of G14, not hardware.

Not done: the functions listed in "AIR features with no AGX lowering", and
running the same cases on both targets in one run (the AIR service is not
reachable from every environment). An ordered atomic reaches AGX as a relaxed one between two
sequentially consistent fences, as AIR expresses it, which costs more than
the AGX backend's own lowering of the ordered atomic.

IR to AGX
----------

Lowering AIR to AGX is designed but not implemented. The design:

* The upstream bitcode reader already parses this backend's ``.air``,
  upgrading its typed pointers to opaque ones, so ``llc -mtriple=agx k.air``
  can read it; no AIR reader is needed.
* An ``agx-lower-air`` module pass, first in the AGX pipeline and a no-op
  without ``!air.kernel``, reads each kernel's argument roles from
  ``!air.kernel``; turns system-value parameters and calls into
  ``llvm.agx.*`` special-register reads; maps AIR math back to LLVM
  intrinsics with an inverse of ``AIRIntrinsicMappings.td`` (generated in the
  AGX target from the same file, with ``Invertible`` excluding
  ``llvm.minimum``/``maximum``); lowers barriers to
  ``llvm.agx.threadgroup_barrier`` between scoped fences, AIR atomics to
  ``atomicrmw``/``cmpxchg``, shuffles, ballots and reductions to their AGX
  intrinsics; and diagnoses AIR functions without an AGX equivalent.
* Address spaces are remapped by cloning with a type remapper: AIR 0
  (thread) becomes AGX 5 (private), 2 (constant) becomes 4, 1 and 3 stay.
  Kernels are rebuilt as ``agx_kernel`` with their buffers in location-index
  order and the system-value parameters removed.
* Validation: the same kernel through AIR on the M2 Max and through AGX on the
  AGX Performance Simulator (``agx_run``, with
  ``llvm/utils/agx-checked/run_checked.py``'s ``launch()``), compared with each
  other and with a CPU reference.
