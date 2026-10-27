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
  ``.air`` produced with ``-air-output=air``: ``inspect`` lists both kernels,
  ``out[tid] = tid + 7`` reads back ``7, 8, 9, 10``, and two ordered dispatches
  over an uploaded buffer read back ``16, 18, 22, 30``.
* No AIR in any test or fixture was produced by Apple's Metal compiler. The
  ``.metallib`` and ``.air`` files in ``Metal.jl/test/metallib`` are built
  from ``.metal`` sources with Apple's toolchain and are deliberately unused.

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

No Rust toolchain, so the AGX Performance Simulator cannot be built
--------------------------------------------------------------------

*What:* the simulator in ``/scratch/AGX-Performance-Simulator`` is a Rust
crate (``agx_run``, ``agx_cost``). The container has no ``cargo``/``rustc``
and no root to install them.

*Impact:* AGX code produced from AIR cannot be executed, so the planned
differential test (same kernel on the M2 Max via AIR and on the simulator via
AGX) needs a toolchain fetched without root. ``llvm/utils/agx-bench`` and
``agx-checked`` are blocked on the same binary.

The GitHub key here is scoped to one repository
-----------------------------------------------

*What:* the SSH key answers ``Hi Jianwen-Ding/llvm-project!``: a deploy key
for that repository alone. It can push this fork but cannot read any other
repository, including the simulator's remote.

AIR features with no AGX lowering
---------------------------------

*What:* the AIR to AGX path translates kernels, buffer arguments, system
values, address spaces, math intrinsics, barriers, atomics, SIMD shuffles,
device-coherent accesses and threadgroup globals. It rejects with a
diagnostic, rather than miscompiling, AIR it has no AGX equivalent for:
SIMD-group matrices (``air.simdgroup_matrix_8x8_*``), SIMD-group async
copies, tensors (``air.*_private_tensor``), indirect argument buffers,
function groups and the 8-operand ``air.atomic.global.cmpxchg.weak.i32``.
