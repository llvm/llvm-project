=============================
Ripple Parallelism Intrinsics
=============================

.. contents::
   :local:
   :depth: 2

Overview
========

**Ripple** is an IR-level mechanism that lets frontends describe
*block shapes* (logical SIMD/tensor extents per processing element) and
lets the backend (through the *Ripple* pass) specialize and
vectorize scalar code based on those shapes. Frontends encode shapes with a
pair of lightweight LLVM intrinsics, while the pass propagates shapes, checks
for consistency, optionally dispatches to *external Ripple functions*
(e.g., math libraries following a naming convention), and can synthesize
specialized, vector-returning clones of scalar functions.

Ripple introduces:

* **Processing Elements (PEs).** A target-specific set of parallel
  resources (e.g. a vector unit) identified by a small integer PE id.

* **Block Shape.** A static (compile-time) tensor shape attached to a PE.
  Shapes are expressed as integer extents; scalars are shapes where all
  extents are ``1``. Shapes across operands broadcast following
  NumPy-style rules, and the broadcasted shape governs parallel
  expansion.

* **Block identity.** The intrinsic :ref:`llvm.ripple.id <ripple-id>`
  exposes the PE index currently considered by the pass; it is the
  anchor the pass uses to propagate parallelism by broadcasting tensor
  shapes across the IR of a function.

* A convention for **external Ripple functions** identified by a *ripple_* prefix
  and optional option tags (e.g., “element-wise” or “masked”). These declarations
  can be auto-loaded from external modules/bitcode and matched against calls to
  route element-wise operations to tensor library implementations.

* A **specialization pipeline** that creates cloned, vector-typed versions of
  functions.

This document specifies the intrinsics, verifier rules, metadata, and
the transformations performed by the Ripple pass.

The Ripple pass runs as a module pass and validates shape usage in the IR,
emitting precise diagnostics when the program violates the rules (e.g., ambiguous
shape sources, or passing a block shape into a declaration-only callee).

Motivation
==========

Modern processors integrate multiple types of processing elements (PEs)—such as
scalar cores, vector units, and sometimes matrix engines—within the same chip.
Efficiently using these resources is challenging because existing programming
models often impose constraints on how scalar and parallel code can be combined.

* **OpenMP(R)** provides a mature model for parallel programming and supports both
  multithreading and vectorization. However, its vectorization is typically
  applied at the loop or whole-function level, which makes it harder to interleave
  scalar and vector operations within the same function.

* **OpenCL(R)** offers fine-grained control and a familiar GPU-style programming
  model, but it usually requires separate host and kernel code, explicit memory
  management, and a different compilation flow, which can increase complexity.

Many kernels are naturally described with implicit SPMD-or-vector semantics (“do
this element-wise over a shape”). Leaving shape knowledge in the front-end
misses interprocedural opportunities and complicates specialization. Ripple
keeps shapes and PE identity **in the IR**:

Why Ripple?
===========

Ripple introduces a compiler-integrated approach that:

* **Allows scalar and PE-specific code to coexist in the same function**: Unlike
  OpenMP's whole-function vectorization, Ripple enables mixing scalar operations
  with vector or matrix operations at a finer granularity.

* **Follows a familiar convention**: Ripple's naming is close to OpenCL and
  CUDA(R), making it more intuitive for developers familiar with GPU programming
  concepts.

* **Supports shape-aware specialization**: Ripple can generate specialized
  versions of functions based on tensor shapes and broadcasting rules.

* **Builds on LLVM infrastructure**: Ripple is implemented as an LLVM pass,
  leveraging the same ecosystem as OpenMP and OpenCL without introducing a new
  runtime or language.

Intrinsic Reference
===================

All Ripple intrinsics live in the ``llvm.ripple.*`` namespace. The
intrinsics are treated as **compile-time shape/control** and are
consumed by the Ripple pass; they are not meant to survive the pass.

.. _ripple-setshape:

``llvm.ripple.block.setshape``
------------------------------

**Syntax**

.. code-block:: llvm
  declare ptr @llvm.ripple.block.setshape(T %pe_id,
                                          T %d0, T %d1, ..., T %d9)

Where:

* ``T`` is any integer type (``llvm_anyint_ty``).

**Semantics**

Creates a *block shape handle* that associates a **Processing Element**
identified by ``%pe_id`` with a static tensor shape whose per-dimension extents
are the remaining arguments. The number of dimensions (``K``) is target/pipeline
dependent. All extents must be positive; ``1`` denotes a degenerate dimension
(scalar along that axis).

.. note::

   The current implementation supports up to 10 dimensions per shape

The *numerical value* of the returned handle is irrelevant; it is used
as a key by the Ripple pass to:

* drive broadcast when aggregating shapes across operands,
* select function specializations and masked variants.

Its use may be extended in the future to carry runtime information.

**Typing/Use**

* Return type is pointer and operand types are any integer ``iX``.
  The handle is a first-class value: it can be SSA-used directly or
  stored/loaded through memory. The pass tracks it through MemorySSA to find the
  defining ``setshape``.
* ``%pe_id`` must be a constant in the function (the pass uses it to
  compute the maximum PE rank and the mapping from PE dimensions to the global
  tensor order).

**Examples**

.. code-block:: llvm

  ; One PE (id 0) with a 1-D vector shape of length 64
  %sh = call ptr @llvm.ripple.block.setshape(i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)

  ; Two-dimensional tile (N=32, M=8) on PE 0
  %tile = call ptr @llvm.ripple.block.setshape(i64 0, i64 32, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)

.. _ripple-getsize:

``llvm.ripple.block.getsize``
-----------------------------

**Syntax**

.. code-block:: llvm

  declare T @llvm.ripple.block.getsize(ptr %block, T %dim)

Where:

* ``T`` is any integer type (``llvm_anyint_ty``).

**Semantics**

Returns the size of dimension ``%dim`` (zero-based, least-significant /
innermost dimension is 0) from the block shape designated by
``%block``. If ``%dim`` is *greater than or equal to* the PE rank,
``1`` is returned (scalar along that axis). An error is raised if the index is
greater than the maximum number of dimensions supported by the Ripple
implementation.

**Typing/Use**

* ``%block`` is the *handle* returned by
  :ref:`ripple.block.setshape <ripple-setshape>` (directly or recovered
  through loads).
* ``%dim`` must be an integer value; non-constant dimensions are
  allowed, but invalid indices are still defined to return ``1``.

**Example**

.. code-block:: llvm

  %tile = call ptr @llvm.ripple.block.setshape(i64 0, i64 32, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %n = call i64 @llvm.ripple.block.getsize(ptr %tile, i64 0) ; -> 32
  %m = call i64 @llvm.ripple.block.getsize(ptr %tile, i64 1) ; -> 8

.. _ripple-id:

``llvm.ripple.id``
------------------

**Syntax**

.. code-block:: llvm

  declare T @llvm.ripple.id(ptr %block, T %dim)

Where:

* ``T`` is any integer type (``llvm_anyint_ty``).

Overview
--------

The ``llvm.ripple.id`` intrinsic returns the index of the current processing
element within a block along a given dimension. This index is used to
distinguish between different elements executing the same code in a block.

Semantics
---------

``llvm.ripple.id`` provides the coordinate of the executing element within the
specified dimension of the block. After the Ripple transformation, this
intrinsic is replaced by a vector of indices ranging from ``0`` to
``llvm.ripple.getsize(%dim) - 1`` for the given dimension.

**Semantics**

The intrinsic is intended as the *anchor* for shape-driven parallel expansion:
starting from a block shape, the pass broadcasts operand shapes, derives masks,
and creates vectorized or if-converted code along the block dimension.

``llvm.ripple.id`` provides the coordinate of the executing element within the
specified dimension of the block. After the Ripple transformation, this
intrinsic is replaced by a tensor (vector) of indices ranging from ``0`` to
``llvm.ripple.block.getsize(%pe_id, %dim) - 1`` along the requested dimension.

Scalar (degenerate) tensor dimensions return the scalar ``0``. Otherwise, the
result is a tensor containing the indices for that dimension.

**Typing/Use**

* ``%block`` is the handle returned by
  :ref:`ripple.block.setshape <ripple-setshape>`.
* ``%dim`` must be an immediate integer value;

* The return type is any integer type.

**Example**

.. code-block:: llvm

  %sh = call i64 @llvm.ripple.block.setshape(i64 0, i64 32, i64 1, ..., i64 1)
  %pe = call i64 @llvm.ripple.id(i64 %sh, i64 0) -> Tensor[0 ... 31]


Rules for Block-Shape Uses
--------------------------

Ripple checks the *use sites* of block-shape values and enforces:

* A use must be able to trace back to a unique ``ripple.block.setshape``. If
  multiple incomparable shapes can reach a use (e.g., divergent stores, PHI with
  different shapes), the pass emits an error pointing at both sources.
* Passing a block-shape value as an argument into a call whose callee is a mere
  declaration (no body to process) is rejected. The frontend must provide a
  definition or use an external Ripple function that the pass can recognize and
  rewrite.
  **Rationale:** This step is essential for Ripple's semantics because shape
  propagation relies on knowing the callee's return shape to compute the
  caller's tensor shape. If the callee is unknown, the program becomes
  ambiguous, preventing safe specialization or vectorization.

.. _ripple-shape-transform-intrinsics:

Ripple Shape-Transforming Intrinsics
------------------------------------

The following intrinsics explicitly **transform a value's Ripple shape**.
They do not necessarily change the LLVM IR *types* of their operands; instead,
they update the shape information tracked by the Ripple pass (used for
vectorization/specialization and legality checks).

These intrinsics compose with the lower-level block-shape primitives:

* :ref:`llvm.ripple.block.setshape <ripple-setshape>`
* :ref:`llvm.ripple.block.getsize  <ripple-getsize>`

**Conventions**

* **Rank and Out-of-Range Dimensions**: If a dimension index is greater than the
  PE rank, that dimension is treated as having extent ``1`` (implicitly
  broadcastable).

* **Broadcast Compatibility Rule**: Two extents are compatible if they are
  equal, or one of them is ``1``. When combining multiple operands, the result
  extent is the maximum of all compatible extents in that dimension.

* **Unambiguous Shape Source**: Any intrinsic that takes a block-shape handle
  requires a **unique** reaching ``llvm.ripple.block.setshape`` at the use site
  (validated with MemorySSA). If multiple different shapes could reach the use
  (via stores/loads/PHIs), the pass rejects the function and emits a diagnostic.

* **IR Identity, Shape Effect**: Unless otherwise noted, the IR *value* and
  *type* produced by these intrinsics are the same as the input value; only the
  **associated Ripple shape** changes. The pass will later materialize the
  appropriate scalar/vector code during transformation and specialization.

.. _ripple-broadcast:

``llvm.ripple.broadcast``
-------------------------

.. code-block:: llvm

  declare T @llvm.ripple.broadcast(ptr %block, i64 %mask, T %x)

Where:

* ``T`` is any scalar integer or floating point type.

**Overview**

Broadcasts ``%x`` across the per-dimension extents encoded by ``%shape`` selected by ``%mask``.
Conceptually, for each dimension index ``i`` set in the mask, the result extent is:

``result_i = if mask & pow2(i + 1)
             then
               max(extent_x_i, extent_shape_i)
             else
               extent_x_i``

subject to the broadcast compatibility rule.

**Arguments**

* ``%block`` - a block-shape handle previously produced by
  :ref:`llvm.ripple.block.setshape <ripple-setshape>`.
* ``%mask`` - a bitset of dimension indices; selecting the dimensions of ``%shape`` to be
  considered by the broadcast
* ``%x`` - the value to broadcast (scalar or first-class value that Ripple tracks).

**Return Value**

The same IR type as ``%x``; the **Ripple shape** is updated to the broadcasted
shape.

**Semantics and Verification**

* Per-dimension broadcast with the compatibility rule (equal-or-1).
* If any dimension is incompatible, the pass emits a broadcast error and
  aborts transformation for the function.
* ``%shape`` must have a **unique** reaching definition; ambiguity is diagnosed.

**Example**

.. code-block:: llvm

  ; Establish a vector length of 8 (rank-1 target)
  %bs = call ptr @llvm.ripple.block.setshape(i64 0, i64 8, i64 1, ..., i64 1)

  ; Broadcast scalar %s to shape [8]
  %y  = call i32 @llvm.ripple.broadcast(ptr %bs, i64 1, i32 %s)


.. _ripple-slice:

``llvm.ripple.slice``
---------------------

.. code-block:: llvm

  declare T @llvm.ripple.slice(T %x, i64 %index1, ..., i64 %index10)

Where:

* ``T`` is any scalar integer or floating point type.

**Overview**

Takes a **slice** of ``%x`` by fixing the dimensions at element ``%index``. The
effect is to set the Ripple extent of that dimension to ``1`` (i.e., collapse
the dimension by extracting the indexed element). The IR type is unchanged; only
the shape is affected.

**Arguments**

* ``%x`` - the value to slice.
* ``%index`` - the element index within that dimension. A value of ``-1`` keeps
  the dimension intact while a positive integer slices (extracts) the element at
  that index.

**Return Value**

Same IR type as ``%x``; the **Ripple shape** has extent ``1`` for positive ``%index``.

**Semantics and Verification**

* Collapses the selected dimension extent to ``1``; other dimensions are
  unchanged.
* Composes with subsequent broadcast and reduction.
* ``%index`` must be integers (``i64``).

**Example**

.. code-block:: llvm

  ; %x currently has shape [8]; slicing lane 3 collapses dim 0 to extent 1.
  %lane = call i32 @llvm.ripple.slice(i32 %x, i64 3, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)


.. _ripple-reduce-family:

``llvm.ripple.reduceX`` Family
------------------------------

**Generic Forms**

.. code-block:: llvm

  declare T @llvm.ripple.reduce.add(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.mul(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.smin(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.smax(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.umin(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.umax(i64 %mask, T %x)
  declare T @llvm.ripple.reduce.and(i64 %mask, T %x)   ; bitwise
  declare T @llvm.ripple.reduce.or (i64 %mask, T %x)   ; bitwise
  declare T @llvm.ripple.reduce.xor(i64 %mask, T %x)   ; bitwise
  declare F @llvm.ripple.reduce.fadd(i64 %mask, F %x)
  declare F @llvm.ripple.reduce.fmul(i64 %mask, F %x)
  declare F @llvm.ripple.reduce.fmin(i64 %mask, F %x)
  declare F @llvm.ripple.reduce.fmax(i64 %mask, F %x)
  declare F @llvm.ripple.reduce.fminimum(i64 %mask, F %x)
  declare F @llvm.ripple.reduce.fmaximum(i64 %mask, F %x)

Where:

* ``T`` is any integer type (``llvm_anyint_ty``).
* ``F`` is any floating-point type (``llvm_anyfloat_ty``).

**Overview**

Reduces ``%x`` along dimension set by ``%mask`` using operator **X**. The
**shape effect** is to collapse the reduced dimension to extent ``1``. On rank-1
targets this typically yields a scalar shape. If ``%mask`` is out of rank, the
reduction is a no-op on shape.

**Arguments**

* ``%x`` - value to reduce. Integer/FP for arithmetic; integer for bitwise.
* ``%mask`` is an integer (recommend i64 to align with other Ripple intrinsics).
  Bit i selects vector dimension i (power-of-two mapping).
  E.g., dim 0 → 1 << 0 = 1, dim 2 → 1 << 2 = 4. Masks can be OR'd (1|4 → dims 0 and 2).

**Return Value**

Same IR type as ``%x``; the **Ripple shape** has extent ``1`` for dimensions in ``%mask``.

**Semantics**

If the logical extent of ``%x`` in ``%mask`` is ``N``, the result is:

.. code-block:: text

  fold_{i=0..N-1}(X, neutral_elem, x[i])

For floating-point reductions, Ripple supposes associativity of floating point
operation. In the future, we would like to support FMF (fast-math flags) on the
call to govern associativity/contracting as usual.

``Floating-Point Variants``

Ripple floating point reduction intrinsics share their semantics with LLVM
vector reduction intrinsics:

* ``ripple.reduce.fmin`` / ``ripple.reduce.fmax`` ->
  ``llvm.vector.reduce.fmin`` / ``llvm.vector.reduce.fmax``
  (IEEE(R) 754-2008 ``minNum`` / ``maxNum`` semantics):

  - If exactly one operand is NaN, the other operand is returned.
  - If both operands are NaN, the result is NaN.
  - Signed-zero ties are defined: ``fmin(+0.0, -0.0) = -0.0``,
    ``fmax(+0.0, -0.0) = +0.0``.
  - Matches C ``fmin`` / ``fmax`` from ``<math.h>``.

* ``ripple.reduce.fminimum`` / ``ripple.reduce.fmaximum`` ->
  ``llvm.vector.reduce.fminimum`` / ``llvm.vector.reduce.fmaximum``
  (IEEE 754-2019 ``minimum`` / ``maximum`` semantics):

  - **NaN-propagating**: if any operand is NaN, the result is NaN.
  - Signed-zero ordering is defined:
    ``fminimum(-0.0, +0.0) = -0.0``,
    ``fmaximum(-0.0, +0.0) = +0.0``.

``Ripple Floating-Point Reduction Semantics``

The following table summarizes the behavior of Ripple intrinsics for key cases,
matching LLVM semantics.

+----------------------+---------------------------+--------------------------------+
| Operation            | ripple.fmin / fmax        | ripple.fminimum / fmaximum     |
+======================+===========================+================================+
| NUM vs qNaN          | NUM                       | qNaN                           |
+----------------------+---------------------------+--------------------------------+
| NUM vs sNaN          | qNaN                      | qNaN                           |
+----------------------+---------------------------+--------------------------------+
| qNaN vs sNaN         | qNaN                      | qNaN                           |
+----------------------+---------------------------+--------------------------------+
| sNaN vs sNaN         | qNaN                      | qNaN                           |
+----------------------+---------------------------+--------------------------------+
| +0.0 vs -0.0         | +0.0(max)/-0.0(min)       | +0.0(max)/-0.0(min)            |
+----------------------+---------------------------+--------------------------------+
| NUM vs NUM           | larger(max)/smaller(min)  | larger(max)/smaller(min)       |
+----------------------+---------------------------+--------------------------------+

Notes:

Ripple intrinsics do not currently model floating-point exceptions.
They share the same behavior as LLVM's unconstrained intrinsics:
operations are performed in the default floating-point environment,
and exception flags (e.g., invalid for signaling NaNs) are ignored.

- Semantics match LLVM intrinsics:
  ``llvm.vector.reduce.fmin/fmax`` and
  ``llvm.vector.reduce.fminimum/fmaximum``.

**Verification**

* ``%mask`` is an integer (recommend i64 to align with other Ripple intrinsics).
  Bit i selects vector dimension i (power-of-two mapping).
  E.g., dim 0 → 1 << 0 = 1, dim 2 → 1 << 2 = 4. Masks can be OR'd (1|4 → dims 0 and 2).
* Operator domain must match types (e.g., bitwise ops require integers).
* The shape associated with ``%x`` must be well-formed (no ambiguous
  block-shape source at this call site).

**Example**

.. code-block:: llvm

  ; %v has shape [8]; sum-reduce along dim 0 → scalar shape [].
  %sum = call i32 @llvm.ripple.reduce.add(i64 1, i32 %v)

.. _ripple-shuffle:

Ripple Shuffle Intrinsics
-------------------------

Ripple exposes two shuffle intrinsics—one for integer elements and one for
floating-point elements—that reorder lanes along the active PE (vector)
dimension. They preserve LLVM IR types and **do not** change Ripple shapes:
only the **lane mapping** along the shuffled dimension is altered.

**Prototypes**

.. code-block:: llvm

  ; Integer elements
  declare T @llvm.ripple.shuffle(T %x, T %y_or_ignored, i1 immarg %pair, ptr %fn)

Where:

* ``T`` is any scalar integer, floating-point or pointer type.
* The **first two operands** have the **same LLVM type** as the return
  (``LLVMMatchType<0>``).
* ``%pair`` is an **immarg i1** (compile-time constant).
* ``%fn`` is a pointer to the **mapping function** (see below).

**Overview**

Let ``E`` be the logical extent of the shuffled dimension (derived from the
Ripple shape associated with the operands at this call site).

* If ``%pair = false`` (single-source):
  * For every output lane ``i`` in ``[0, E)``, compute ``j = fn(i, N)`` and set
    ``result[i] = x[j]``.
  * The second argument ``%y_or_ignored`` is **ignored** (you may pass ``undef``,
    ``poison``, or simply reuse ``%x``).

* If ``%pair = true`` (two-source):
  * For every output lane ``i`` in ``[0, E)``, compute ``j = fn(i, N)`` in the
    range ``[0, 2E)``. If ``j < E``, the element comes from ``x[j]``; otherwise,
    it comes from ``y[(j - E)]``.
  * Both ``x`` and ``y`` must be available and have compatible shapes (see below).

In both cases, the **Ripple shape is preserved** (extent remains ``E`` in the
shuffled dimension). If ``E == 1``, the shuffle is a no-op.

**Mapping Function**

``%fn`` is a pointer to a pure function that maps **output lane index** to
**origin lane index**, and now receives the **tensor size**:

.. code-block:: llvm

  ; Canonical signature (two arguments)
  declare i64 @ripple.shuffle.map(i64 %out_index, i64 %tensor_size)

Parameters:

* ``%out_index`` - the output lane index ``i`` in ``[0, E)``.
* ``%tensor_size`` - the total number of logical elements **N** in the value,
  i.e. the **product of all per-dimension extents** at this call. On rank-1
  targets, ``N == E`` (the vector length).

Requirements:

* The function is assumed **willreturn** (as per intrinsic attributes). Frontends
  should mark it **readnone**/**nosync** where applicable.
* For ``%pair = false`` it must return ``j`` in ``[0, E)``.
* For ``%pair = true`` it must return ``j`` in ``[0, 2E)`` (concatenation
  ``x`` then ``y``).
* Duplicate indices (gather) are **allowed**. There is no separate permutation-only
  mode; backends may still recognize pure permutations.

**Shape and Compatibility**

* **Rank and Dimension**: The shuffled dimension is the active PE dimension
  (rank-1 vector dimension in the current machine model). If that dimension's
  extent is ``1``, shuffling has no effect.

* **Two-source mode** (``%pair = true``):
  * ``x`` and ``y`` must be shape-compatible along the shuffled dimension.
    The default requirement is **equal extent** in that dimension.
  * If a more permissive broadcast (equal-or-1) is supported, it must be
    implemented consistently by the pass; otherwise the pass diagnoses a mismatch.

* **Single-source mode** (``%pair = false``): only ``x`` is used; the second
  operand is ignored.

**Return Value**

* Same **LLVM type** as the inputs.
* **Ripple shape** is unchanged; only the lane mapping in the shuffled dimension
  is modified.

**Verification**

* ``%pair`` must be an **immarg** constant.
* ``%fn`` must be analyzable (available to the pass) and match the canonical
  signature ``i64(i64, i64)`` or be bitcastable thereto.
* For ``%pair = true``, the pass checks that the shapes of ``x`` and ``y`` are
  compatible as required (typically equal extent along the shuffled dimension).

**Diagnostics**

* If the mapping function returns an index out of the allowed range
  (``[0, E)`` in single-source, ``[0, 2E)`` in two-source) a diagnostic is
  issued.
* Ambiguous or incompatible shapes are diagnosed by the pass.

**Examples**

Single-source reverse (rank-1):

.. code-block:: llvm

  ; With rank-1, %tensor_size == E.
  ; Reverse: j = (tensor_size - 1 - i).
  define i64 @rev(i64 %i, i64 %N) readnone willreturn {
  entry:
    %Nm1 = add i64 %N, -1
    %j   = sub i64 %Nm1, %i
    ret i64 %j
  }

  ; Shuffle usage (pair = false). Second operand is ignored; pass %x again.
  %y = call i32 @llvm.ripple.ishuffle(i32 %x, i32 %x, i1 false, ptr @rev)

Two-source interleave (rank-1):

.. code-block:: llvm

  ; Interleave x and y: result[2*k] = x[k], result[2*k+1] = y[k].
  ; For rank-1, %N == E, and 2E is the concatenation length.
  define i64 @interleave_map(i64 %i, i64 %N) readnone willreturn {
  entry:
    %isodd = and i64 %i, 1
    %half  = lshr i64 %i, 1
    %j1    = add i64 %N, %half           ; indexes into y half when odd
    %sel   = select i1 (icmp ne i64 %isodd, 0), i64 %j1, i64 %half
    ret i64 %sel
  }

  ; Interleave two vectors x and y (same extent along shuffled dimension)
  %z = call float @llvm.ripple.fshuffle(float %x, float %y, i1 true, ptr @interleave_map)

**Notes and Best Practices**

* Keep ``%fn`` simple and **pure** (``readnone willreturn``) to maximize
  constant-folding and lowering opportunities.
* In **single-source** mode, it is fine to pass ``undef`` or reuse ``%x`` as the
  second operand; the pass **ignores** it when ``%pair = false``.
* On **rank-1 targets**, ``%tensor_size`` equals the shuffled dimension's extent
  (so examples can use it directly as the lane count).
* If you need strict permutation properties for codegen, implement a bijective
  mapping in ``%fn``; the intrinsic interface does not enforce it but backends
  will often pattern-match permutations for efficient lowering.

Composition Rules
-----------------

1. **Order of Effects** - Intrinsics apply in program order. Broadcasting then
   slicing produces a single lane of the broadcasted tensor; slicing then
   reducing typically yields the selected element (scalar shape on rank-1).

2. **Elementwise Uses** - When these intrinsics feed elementwise operations or
   elementwise external calls, Ripple recomputes the **broadcasted** shape
   across all operands using the compatibility rule. Incompatibilities are
   diagnosed with a message that lists operand shapes.

3. **Out-of-Range Dimensions** - For ``%dim >= rank``, both **slice** and
   **reduce** act as shape no-ops (equivalent to operating on a size-1
   dimension). This mirrors how querying a too-large dimension via
   :ref:`llvm.ripple.block.getsize <ripple-getsize>` yields ``1``.

4. **Ambiguous Block Shapes** - Any use that (directly or via memory) depends on
   more than one possible ``block.setshape`` is rejected with an explicit
   diagnostic pointing to both sources.

**Worked Example**

.. code-block:: llvm

  ; 1) Establish a vector shape of length 8 for PE #0
  %bs  = call ptr @llvm.ripple.block.setshape(i64 0, i64 8, i64 1, ..., i64 1)

  ; 2) Broadcast a scalar %s to [8]
  %b   = call i32 @llvm.ripple.broadcast(ptr %bs, i64 1, i32 %s)

  ; 3) Slice lane 3 (dim 0), shape collapses to []
  %lane = call i32 @llvm.ripple.slice(i32 %b, i64 3)

  ; 4) Reduce %b (shape [8]) to scalar with addition (i.e., 8 * %s)
  %sum  = call i32 @llvm.ripple.reduce.add(i32 %b, i64 1)


**See Also**

* :ref:`llvm.ripple.block.setshape <ripple-setshape>`
* :ref:`llvm.ripple.block.getsize  <ripple-getsize>`


Tensor Shapes, Broadcasting, and Element-Wise Calls
===================================================

Ripple tracks a *tensor shape* (vector extent per dimension) for IR values and
functions. When calling **element-wise external Ripple functions**, all vector
arguments must either be equal in length or broadcast-compatible (scalars and
tensors can broadcast). The computed broadcasted shape drives the result shape
and, when masking is involved, the mask vector width as well.

If operands cannot be broadcast together, the pass emits a diagnostic on the
call site.

External Ripple Functions
=========================

Overview
--------

An *external Ripple function* is a normal LLVM function declaration whose **name
uses the Ripple prefix** (and optional option tags). Such declarations can live
in your module or be **auto-loaded** by the pass from additional modules/bitcode
files listed via the pass' library list; matching declarations are cloned into
the current module and registered as candidates.

Name and Options
----------------

* The pass recognizes functions whose name starts with the **Ripple prefix**
  (``ripple_``) and then strips a sequence of known **option tags**; the
  remaining tail is the function's **base scalar name** (e.g., a libm symbol).
* Recognized options include:

  * **Element-wise** (``_ew``) — marks the function as operating element-wise across
    vector arguments; all vector arguments (and the vector return, if any) must
    share the same element count, or scalars are broadcast.
  * **Masked** (``_mask``) — indicates the function accepts an additional *tensor mask*
    argument (last parameter, adjusted if ``sret`` is present). The mask must be
    a vector of an integer scalar type (e.g., ``<N x i1>``, ``<N x i8>``).
  * **Pure** (``_pure``) — asserts the function is pure except for ``sret``/``byval``
    argument effects, enabling safe masked expansion. Even without this option,
    the pass deems a function “pure enough” if it is ``readnone`` or if its
    memory effects are restricted to argument pointees and those pointees are
    only ``sret``/``byval`` *non-pointer* values.

Type Normalization
------------------

When comparing a call with a candidate external Ripple function, the pass
*normalizes* both sides to their **true** signature:

* ``sret``: treat as returning the pointee type instead of a pointer argument.
* ``byval``: treat as passing the pointee value type instead of a pointer.

The pass also verifies element types (ignoring vector widths for element-wise
matching) and ensures vector element counts are consistent in the element-wise
case. Scalar/aggregate arguments are allowed as long as the option and matching
rules are satisfied.

Matching and Libm Mapping
-------------------------

For some LLVM FP intrinsics, Ripple derives a **lib function name** (via
``TargetLibraryInfo`` w/ extensions) to look up a corresponding
external Ripple function by scalar base name (post-prefix/options). This covers
intrinsics like ``sqrt``, ``sin``, ``exp``, ``log``, etc. When a match is found,
the pass rewrites the call to the external Ripple declaration (preferring an
unmasked variant when mask is not required).

Portability & Name Mangling of External Ripple Functions
--------------------------------------------------------

Current matching of *external Ripple functions* relies on two mechanisms:

1. **Naming convention**: a Ripple prefix plus option tags, followed by the
   scalar **base name** of the function (e.g., a libm symbol).
2. **Type normalization** via IR attributes:
   * treat ``sret`` as a return-in-argument (extract the pointee type);
   * treat ``byval`` as pass-by-value of the pointee type.

This lets Ripple compare *element types* and check shape compatibility without
depending on ABI lowering side-effects. However, this approach **depends on the
frontend/target emitting informative IR attributes**. On some targets—most
notably **AArch64**—vector arguments are typically passed in registers according
to the ABI and **may not be annotated via ``sret``/``byval`` in ways that help
Ripple infer vector/tensor types**. In practice, this can make external function
matching brittle or target-dependent.

.. note::

   The meaning of ``sret`` and ``byval`` is target/ABI-sensitive; their presence,
   absence, or exact lowering are not guaranteed to encode vector/tensor types in
   a portable way. See the LLVM language/reference and discussions for details on
   these attributes and their ABI impact.

**Planned Direction: Name + Type Mangling (C-Level)**

External Ripple functions already use C-compatible linkage. Today, matching
relies on a naming convention plus `sret`/`byval` attributes to infer argument
and return types. This works on many targets but is fragile on ABIs that do not
use these attributes for vector arguments (e.g., AArch64).

To make matching robust and portable, we plan to introduce **full name + type
mangling at the C symbol level** in the future. This will encode the **element
type** and **tensor extents** directly in the symbol name, avoiding reliance on
ABI-specific attributes.

Proposed format:

.. code-block:: text

   ripple_<options>_<scalarbase>__<ret>_<arg0>_<arg1>_...
   where:
     <ret>  = return type encoded as <elem>t<d0>[x<d1>...]
     <argN> = argument type encoded as <elem>t<d0>[x<d1>...]
     <elem> ∈ { f16, f32, f64, i8, i16, i32, i64, ... }
     <d0>.. <dk> are positive integers (tensor extents, row-major by convention)

Here, the signature encodes the return type first, followed by each argument
type, similar to C++ name mangling but in a simplified, human-readable form.

Examples:

* **Element-wise**, unmasked sine returning `float[32][2]` and taking one
  `float[32][2]` argument:

  .. code-block:: c

     // C symbol name:
     ripple_ew_pure_sin__f32t32x2_f32t32x2
     // Declaration:
     float ripple_ew_pure_sin__f32t32x2_f32t32x2(float, float);

* **Masked**, element-wise power returning `double[8]` and taking two
  `double[8]` arguments plus a mask:

  .. code-block:: c

     ripple_masked_ew_pure_pow__f64t8_f64t8_f64t8_i1t8
     double ripple_masked_ew_pure_pow__f64t8_f64t8_f64t8_i1t8(double, double, _Bool mask[8]);

* **Non-element-wise**, add returning `i32[4][4]` and taking two `i32[4][4]`
  arguments:

  .. code-block:: c

     ripple_pure_add__i32t4x4_i32t4x4_i32t4x4
     int ripple_pure_add__i32t4x4_i32t4x4_i32t4x4(int, int);


**Important:** This scheme will remain fully compatible with the C calling
convention so that external libraries can implement these functions without
LLVM-specific knowledge.

**Migration strategy (future):**

1. **Dual support**: accept both current (prefix + options + base name +
   `sret`/`byval`) and fully-mangled names during a transition period.
2. **Prefer mangled names** when both forms exist.
3. **Diagnostics**: warn when matching depends on ABI attributes on targets
   where such inference is unreliable.

.. note::

   The current approach (naming + `sret`/`byval`) is sufficient for our
   immediate use cases. However, to support other targets such as **AArch64**
   and architectures with **scalable vectors** (e.g., Arm(R) SVE, RISC-V(R) V), this
   mangling scheme will become necessary. Implementing it will require invasive
   changes to the Ripple pass and external libraries, so it is planned for a
   later phase.

Specialization Pipeline
=======================

When shape information indicates that a scalar callee can be specialized to a
vector signature, Ripple can synthesize *specializations* of that function:

* **Pending specialization** clones the original function and duplicates its
  parameter list: the first half are the original arguments, the second half
  are the “Ripple tensor arguments” (vector-promoted based on shapes). The pass
  then completes and transforms this clone, finally producing the **final**
  specialization.
* Final specialized functions are named with a fresh
  ``"ripple.specialization.final."`` prefix (plus a unique counter) and the original
  function name. A **masked** companion declaration is created with name
  ``"masked." + final_specialization_name`` and an extra trailing mask argument.
* For **void-returning** functions, the pass may **pre-declare** the final
  specialization early to break cycles; it records the final name via the
  ``"ripple.specialization.final.fname"`` metadata on the pending clone and then
  finishes cloning later.

Return and Argument Types
-------------------------

* If the specialized return shape is a vector, the pass vectorizes the return
  type and **drops** scalar-only return attributes that do not apply to vectors
  (e.g., ``zext``/``sext``/``inreg``/``noundef``). If the return shape is scalar,
  the original return type is kept.
* The **mask** type for whole-function masking uses the element type ``i1`` and a
  vector width equal to the broadcast of the argument shapes.

Function Call Resolution Strategy
---------------------------------

When Ripple processes a call site, it applies the following resolution order:

1. **External Ripple Function**
   If a matching external Ripple function exists (based on the Ripple naming
   convention and type checks), the call is rewritten to invoke that external
   function. This is the preferred path for element-wise operations and
   library-provided implementations.

2. **Specialization of Known Definition**
   If the callee has a definition available in the current module and is
   eligible for specialization, Ripple generates a specialized version of the
   function with vectorized argument and return types. The call is then updated
   to target this specialized function.

3. **Scalar Fallback (Loop Expansion)**
   If neither an external Ripple function nor a specialization is possible,
   Ripple assumes the function is only available in scalar form. In this case,
   Ripple extracts each element from the vector arguments, calls the scalar
   function in a loop, and constructs a new vector from the scalar results.

This order ensures that the most efficient implementation is chosen when
available, while preserving correctness when only scalar definitions exist.

Shape Metadata
==============

The pass uses the function-level metadata key
``"RippleFunctionShapeMetadata"`` to record shapes:

* The node contains *``N + 1`` operands* for a function with ``N`` arguments:
  one tensor-shape metadata per argument followed by the return shape. For
  ``void`` functions, the return shape is the scalar shape.

* Shapes are encoded in constant metadata nodes; These nodes are internal-only
  and are dropped at the end of the ripple module pass.

Additionally, pending specializations of void functions carry a
``"ripple.specialization.final.fname"`` metadata node holding the name of the
pre-declared final specialization. This metadata allows the Ripple pass to
process other functions and resolve call sites before the actual definition of
the specialization is generated, which is essential for handling cycles and
void-returning functions during specialization.

Diagnostics
===========

Ripple emits diagnostics with source locations when available:

* **Ambiguous block shape** — multiple definitions can reach a query; the pass
  points at both sources.
* **Missing block shape** — a use of a block-shape value cannot be traced back to
  a ``ripple.block.setshape``.
* **Illegal call with block shape** — passing a block-shape value into a
  declaration-only function (no body) is rejected. Provide a definition or use
  an external Ripple function declaration the pass can rewrite.
* **Broadcasting failure** — operands to an element-wise or specialization
  decision are not broadcast-compatible.

The pass also reports non-fatal **warnings** when loading external libraries
fails or when a duplicate Ripple symbol with a type mismatch is encountered in
the current module.

Worked Examples
===============

Defining and Querying a Block Shape
-----------------------------------

.. code-block:: llvm

   ; Suppose PE 0 runs with a 1-D logical vector of length %N.
   %N     = ... ; i64
   %bs    = call i64 @llvm.ripple.block.setshape.i64(i64 0, i64 %N)

   ; Later, query the active size of dimension 0:
   %len0  = call i64 @llvm.ripple.block.getsize.i64(i64 %bs, i64 0)
   ; If we queried a higher dimension, the result would be 1.

The pass will validate that ``%bs`` is the unique block-shape reaching the
``getsize`` use, even if the value was stored to and later loaded from memory
(e.g., at ``-O0``). Ambiguity triggers an error.

Element-Wise External Function
------------------------------

.. code-block:: llvm

   ; External Ripple declaration for element-wise sinf
   ; (exact option tags/prefix come from the Ripple naming convention).
   declare <4 x float> @"ripple_ew_sinf"(<4 x float>)

   ; Scalar IR, but with a vector shape known to Ripple:
   ; %v: <4 x float> logical shape (e.g., via a prior setshape)
   %r = call float @sinf(float %x)

If TargetLibraryInfo provides a matching symbol for the intrinsic/base name,
Ripple can rewrite the call to the external Ripple declaration and pass a
vector-typed argument/return according to the known shape. Masked variants are
preferred only when a mask is required.

Function Specialization
-----------------------

When calling a *scalar* function with vector-shaped arguments, Ripple may create
a specialized clone:

.. code-block:: text
   ripple.specialization.<id>.<orig_name>
   masked.ripple.specialization.<id>.<orig_name>

The specialized function has vector argument/return types that match the
broadcasted shapes, and a masked sibling (with trailing ``<N x i1>``) for
whole-function masking. The pass attaches ``RippleFunctionShapeMetadata`` to the
specialized declarations/definitions, name mangling is not necessary.

Pass Usage and Interaction
==============================

Ripple is exposed to users as a module pass. Invoke it with:

.. code-block:: bash

   opt -passes="ripple" input.ll -o output.ll

The module pass is responsible for:

- Validating all Ripple intrinsics across the module.
- Loading and registering external Ripple function declarations.
- Managing specialization dependencies and finalization.
- Running internal per-function transformations in a safe order.

This orchestration is essential for correctness: running Ripple per function in
isolation would break shape propagation and specialization.

Implementation Notes (for Contributors)
=======================================

* The pass relies on **MemorySSA** and **AA** to trace block-shape definitions
  through stores/loads and to determine clobbering/aliasing behavior. Only
  **must-alias** clobbers are considered; ambiguous flows are rejected.
* External Ripple functions are registered from the current and auxiliary
  modules; their candidacy is decided by naming (prefix/options), **type
  normalization** (accounting for ``sret``/``byval``), vector width checks in
  element-wise mode, and optional **mask** argument validation.
* Specializations reuse utilities to clone functions, fix up metadata, and
  record final names/links with the ``"ripple.specialization.final.fname"``
  metadata. A helper creates a temporary mask for whole-function masking during
  transformation, later replaced by a formal mask parameter in the masked
  specialization.

---
Arm is a registered trademark of Arm Limited (or its subsidiaries).

CUDA is a registered trademark of NVIDIA Corporation.

OpenCL is a registered trademark of Apple Incorporated.

OpenMP is registered trademark of the OpenMP Architecture Review Board

RISC-V is a registered trademark of RISC-V International.

IEEE is a registered trademark of the Institute of Electrical and Electronics Engineers, Inc.
