=======================
AIR Functionality Index
=======================

.. contents::
   :local:

This page lists the AIR functionality that the open-source AIR producers use,
and its state in this backend. It was compiled from the sources of:

* **dxmt** (``src/airconv``): DXBC/DXIL to AIR, with ``nt/air_builder.cpp``
  building every AIR call;
* **floor** and **floor_llvm** (``include/floor/device/backend/metal*.hpp``,
  ``lib/Transforms/LibFloor/MetalFinal.cpp``): C++ to AIR;
* **Metal.jl** (``src/device/intrinsics``): Julia to AIR.

No output of Apple's Metal compiler was used. Signatures come from those
sources; behaviour comes from running this backend's output on an Apple M2
Max through the AIR service, with ``llvm/utils/air-gpu`` (the group named in
the last column).

States:

* **Verified**: produced by this backend (or passed through from IR that
  calls it) and checked on the device against a reference.
* **Lowered**: produced from LLVM IR by this backend, by the same rule as
  verified entries, but not run on its own.
* **Passed through**: a call in the input is written out as it is; not run,
  because the AIR service only dispatches compute kernels on buffers.
* **Unsupported**: diagnosed or known not to work.

Naming
======

AIR functions are named ``air.<name>.<suffix>``. The suffix is the overload
type: ``f32``, ``f16``, ``i32``, ``v4f32``, ``p1f32`` (a pointer to device
``float``)... Where the result depends on signedness the suffix starts with
``s.`` or ``u.`` (``air.min.s.i32``), and floats then use ``f.``
(``air.convert.f.f32.s.i32``). Relaxed math functions are ``air.fast_<name>``,
for ``float`` only.

LLVM intrinsics
===============

``llvm-to-air-intrinsics`` lowers LLVM intrinsics by this rule
(``AIRIntrinsicMappings.td``); others are passed to Metal unchanged.

================================== ======================================= =========================
LLVM                               AIR                                     State
================================== ======================================= =========================
``fabs floor ceil trunc rint``     ``air.<same>``                          Verified (llvm-math-f32)
``round sqrt``
``exp exp2 exp10 log log2 log10``  ``air.<same>``                          Verified (llvm-math-f32)
``sin cos tan asin acos atan``     ``air.<same>``                          Verified (llvm-math-f32)
``atan2 sinh cosh tanh pow``
``minnum maxnum``                  ``air.fmin``, ``air.fmax``              Verified (llvm-math-f32)
``minimum maximum``                ``air.fmin/fmax`` after NaN handling    Lowered
``fma``                            ``air.fma``                             Verified (llvm-math-f32)
any of the above with ``afn``      ``air.fast_<name>`` (f32)               Verified (llvm-math-afn)
half and vector overloads          same name, ``.f16``, ``.v4f32``...      Verified (types)
``smin smax umin umax``            ``air.min/max.s/u``                     Verified (llvm-int)
``abs ctpop ctlz cttz``            ``air.abs.s popcount clz ctz``          Verified (llvm-int)
``bitreverse``                     ``air.reverse_bits``                    Verified (llvm-int)
``copysign``                       unchanged; no ``air.copysign`` exists   Verified (llvm-math-f32)
``memcpy memset``                  unchanged                               Verified memcpy (memory)
``uadd.sat`` and family            unchanged (floor passes them through)   Passed through
``vector.reduce.*``, ``memmove``   not lowered; floor expands them first   Unsupported
================================== ======================================= =========================

AIR math called directly
========================

* All of the above, precise and ``fast_``: verified (air-math-f32).
* ``rsqrt``, ``fract``, ``sinpi``, ``cospi``, ``tanpi``, ``asinh``,
  ``acosh``, ``atanh``, ``fmod``: verified (air-math-f32).
* ``sincos``, returning the cosine through a pointer: verified (extra).
* ``dot``: verified (extra).
* ``mul_hi.s``/``mul_hi.u``: verified (misc).
* ``convert``: verified for int to float (extra).
* ``pack``/``unpack``, ``unorm``/``snorm``, 4x8 and 2x16: verified for
  ``pack.unorm4x8`` (extra).
* ``powr``, ``nextafter``, ``fast_clamp``, ``mad_sat``, and ``clz``, ``ctz``,
  ``popcount``, ``reverse_bits``, ``abs``, ``min``, ``max`` called directly:
  passed through (the LLVM intrinsics for the latter are verified).

Compute system values
=====================

All eleven ``[[...]]`` compute attributes, called as ``air.<name>`` in the
input and turned into kernel parameters with their roles, are verified
(sysvals): ``thread_position_in_grid``, ``thread_position_in_threadgroup``,
``threadgroup_position_in_grid``, ``threads_per_threadgroup``,
``threadgroups_per_grid``, ``threads_per_grid``,
``thread_index_in_threadgroup``, ``thread_index_in_simdgroup``,
``simdgroup_index_in_threadgroup``, ``threads_per_simdgroup``,
``simdgroups_per_threadgroup``. dxmt's OpenCL-style
``air.get_global_id/get_local_id/get_group_id(i32 dim)`` are passed through
and verified (extra).

SIMD-group and quad-group functions
===================================

* ``simd_shuffle`` by index, xor, up and down, and ``quad_shuffle_xor``:
  verified (simd).
* ``simd_shuffle_and_fill_down``: verified (extra).
* ``simd_sum``, ``simd_min``, ``simd_max``, ``simd_and``, ``simd_or``,
  ``simd_xor``, and inclusive and exclusive prefix sums: verified (simd).
* ``simd_ballot``, ``simd_vote_any``, ``simd_vote_all``: verified (simd,
  extra).
* ``quad_ballot``: verified (extra).
* ``quad_shuffle`` by index, up and down, ``*_and_fill_up``, ``quad_vote_*``,
  ``simd_product``: passed through.
* 8x8 SIMD-group matrices (``simdgroup_matrix_8x8_load``, ``store``,
  ``multiply_accumulate``, ``init_filled``, AIR 2.8 form): verified in f32
  and f16, with accumulation, a transposed operand and threadgroup memory
  (matrix).
* Tensors (``air.*_private_tensor``, cooperative tensors): passed through;
  they need MSL 4 tensor support and are unverified.

Synchronization and atomics
===========================

* ``air.wg.barrier(flags, scope)``: verified (threadgroup).
* ``air.wg.barrier(flags, order, scope)``, the AIR 2.9 form: verified
  (threadgroup); the device accepts it in AIR 2.8 modules.
* ``air.simdgroup.barrier``: verified (extra).
* ``air.atomic.fence``, from LLVM ``fence``: verified (atomics).
* ``air.atomic.global``/``local`` operations, from ``atomicrmw``: add, sub,
  and, or, xor, signed and unsigned min and max, xchg, float add: verified
  (atomics, threadgroup).
* ``air.atomic.*.cmpxchg.weak``, from ``cmpxchg``: verified (atomics).
* Atomic loads and stores: verified (atomics).
* ``cmpxchg.weak`` with a memory-flags operand (MSL 4.1): lowered; the
  device's OS only has MSL 4.0.
* 64-bit atomics (``atomic_ulong`` min and max): unsupported.

Memory
======

Device, constant, threadgroup and private memory, structs and nested
aggregates, pointer ``phi`` and ``select``, threadgroup arrays, constant
tables and constant buffer arguments are verified (memory, threadgroup).
``air.load.device_coherent``/``air.store.device_coherent`` are passed through
and verified (misc). Argument buffers (``air.indirect_buffer``,
``air.argbuf.*``) are produced by ``air-argument-buffer`` for Triton's
convention but not run.

Not available through the AIR service
=====================================

The service runs compute kernels on buffers, so these are passed through but
cannot be checked here. The backend has no lowering of its own for any of
them; front ends call them directly.

* Textures and samplers: ``air.sample_*``, ``air.sample_compare_*``,
  ``air.gather_*``, ``air.read_*``, ``air.write_*``, texture atomics
  (``air.atomic_*``), ``air.calculate_[un]clamped_lod_*``, ``air.get_*``
  queries, ``air.fence_*`` (texture fences).
* Graphics stages and their attributes: vertex (``air.vertex_id``,
  ``air.instance_id``, ``air.base_vertex``, ``air.base_instance``, vertex
  inputs and outputs, ``air.position``, ``air.clip_distance``,
  ``air.point_size``), fragment (``air.discard_fragment``,
  ``air.front_facing``, ``air.sample_id``, ``air.sample_mask``,
  ``air.point_coord``, ``air.barycentric_coord``, ``air.render_target``,
  ``air.depth``, ``air.stencil``, ``air.raster_order_group``), interpolation
  (``air.interpolate_*``), tessellation (``air.patch_*``,
  ``air.position_in_patch``), mesh and object stages (``air.set_*_mesh``,
  ``air.mesh_*``, payloads).
* Function constants (``air.function_constants``,
  ``air.is_function_constant_defined``), visible and intersection functions,
  ``air.os_log``, ``air.source_file_name`` and other debug metadata.
