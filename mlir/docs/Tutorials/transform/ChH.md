# Chapter H: Reproducing Halide Schedule

This chapter demonstrates how a schedule from the [Halide
DSL](http://halide-lang.org) can be implemented using transform dialect for
structured ops.

Note that the IR below is pseudo-code with types removed for brevity. It may
also get out of sync with the current syntax. Always refer to the source code in
[mlir/examples/transform/ChH](https://github.com/llvm/llvm-project/tree/main/mlir/test/Examples/transform/ChH)
as the source of truth.

## Channeled Convolution

The Transform dialect provides a substrate for implementing “transformation
directive” domain-specific languages (DSLs) in MLIR. Such a DSL, at least in its
scheduling part, can target the operations in the Transform dialect that are
later applied by the compiler. Sets of transform operations, or even new
dialects leveraging the same interfaces and infrastructure, can be added to
support a specific DSL for a particular scheduling model. In this chapter, we
will revisit the Halide DSL that has (re)popularized separate specification of
schedules originally for image processing programs.

Two approaches Halide to the Transform dialect are possible:

*   Create a new dialect that corresponds to the computational part of Halide
    DSL, and define a set of transformations wrapped into Transform dialect
    operations, that correspond to the scheduling part of the DSL.
*   Map the Halide abstractions to the existing MLIR abstractions, for both
    parts of the DSL.

We will consider the latter approach as the computational part of the DSL easily
maps to the structured ops in the Linalg dialect. This also gives us the
opportunity to discuss how Linalg transformations on the so-called structured
operations are similar to or different from the existing transformations.

We will consider the 2D channeled convolution example extracted from Halide
[application
examples](https://github.com/halide/Halide/tree/294f80c49bf3bb8582446613c25fcce03b82bcd8/apps/conv_layer).

```cpp
// Sizes of the problem.
const int N = 5, CI = 128, CO = 128, W = 100, H = 80;

// Sized inputs. Note that the order of dimensions is
// inverted in Halide with respect to C++, so the last dimension
// in the list (N for input, CI for filter) is the least
// frequently varying. The C++ equivalent is input[N][H+2][W+2][CI].
Buffer<float, 4> input({CI, W+2, H+2, N}, "input");
Buffer<float, 4> filter({CO, 3, 3, CI}, "filter");
Buffer<float, 1> bias(std::vector<int>{CO}, "bias");

// ... data initialization happens here ...

// Declarations of "mathematical functions" for convolution and relu.
Func conv("conv"), relu("relu");

// Iterators/subscripts.
Var x("x"), y("y"), c("c"), n("n");

// 3D reduction domain (channels and 2 window dimensions),
// dimensions are later referred to as r.x, r.y, r.z.
RDom r(0, CI, 0, 3, 0, 3);

// Core convolution with the result initialized to the bias value.
// Note that the order of iterators is inverted in Halide DSL,
// i.e. `n` corresponds to the lest frequently-varying (outermost) dimension
// here and below.
conv(c, x, y, n) = bias(c);
conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);

// ReLU rectification, an elementwise operation.
relu(c, x, y, n) = max(0, conv(c, x, y, n));
```

This can be almost directly converted to Linalg dialect operating on tensors,
which is conceptually closer to the “mathematical function” abstraction and is
where the majority of transformations are available.

```mlir
// Bias. Using a named Linalg operation for brevity.
%bias_init = tensor.empty() : !toutput
%biased = linalg.broadcast ins(%bias : !tbias)
                          outs(%bias_init : !toutput) dimensions = [0, 1, 2]

// Convolution proper. While Linalg has named operations for 2D convolutions,
// the one in the Halide example has an uncommon order of filter dimensions
// and is not supported. It also takes the filter as first argument. This
// code recreates it faithfully using the generic form.
%convolved = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel", "parallel",
                    "reduction", "reduction", "reduction"],
  indexing_maps = [
    affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
    affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
    affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
  ]
} ins(%filter, %input: !tfilter, !tinput)
  outs(%biased : !toutput) {
^bb0(%in: f32, %f: f32, %b: f32):
  // Note the fastmath attributes that allow operations to be recombined into
  //   %0 = math.fma %in, %f, %b : f32
  // later on and to reorder reductions.
  %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
  %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
  linalg.yield %0 : f32
} -> !toutput

// ReLU is just a max(0, x).
%c0 = arith.constant 0.0 : f32
%relued = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> ()>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ]
} ins(%c0, %convolved : f32, !toutput)
  outs(%output : !toutput) {
^bb0(%cst: f32, %in: f32, %out: f32):
  %0 = llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32
  linalg.yield %0 : f32
} -> !toutput
```

In Halide, a function such as `conv` may consist of two parts: a “functional”
initialization computation and an in-place update for reductions. This is
expressed as two C++ statements in the embedded DSL, but internally is
represented in a single object. Linalg doesn’t have such a capability to the
initialization and the update are represented as two distinct Linalg operations
that are not connected to each other. Furthermore, the `x`, `y`, `c`, `n`
variables in Halide DSL correspond to implicit loops iterating over the
corresponding objects, which implies that functions sharing these variables in
their definitions also share the corresponding loops. In other words, the loop
equivalent of the Halide definition starts in a fully-fused form. The Linalg
model is the opposite with each structured operation corresponding to its own
loop nest, resulting in a fully-distributed form. This will affect how the
schedule is constructed later on.

The loop structure for Halide computation resembles the following (adapted from
debug dump with `HL_DEBUG_CODEGEN=1`)

```python
for n
  for y
    for x
      for c
        conv[n, y, x, c] = bias[c]
        for rz
          for ry
            for rx
              conv[n, y, x, c] += filter[rx, rz, ry, c] * input[n, y+rz, x+ry, rx]
        relu[n, y, x, c] = max(0, conv[n, y, x, c])
```

The loop structure for the Linalg computation is as follows (obtained by
`mlir-opt --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor
--one-shot-bufferize --convert-linalg-to-loops`)

```python
for n
  for y
    for x
      for c
        init[n, y, x, c] = bias[c]
for n
  for y
    for x
      for c
        for rz
          for ry
            for rx
              conv[n, y, x, c] += filter[rx, rz, ry, c] * input[n, y+rz, x+ry, rx]
for n
  for y
    for x
      for c
        relu[n, y, x, c] = max(0, conv[n, y, x, c])

```

## Mapping Halide Scheduling Primitives to Linalg Structured Transforms

The complete Halide schedule listed in the example is as follows

```cpp
Var co, ci, xo, xi;
relu.split(c, co, ci, vec * tile_w)
  .split(x, xo, xi, tile_h)
  .reorder(ci, xi, xo, y, n, co)
  .vectorize(ci, vec)
  .unroll(ci)
  .unroll(xi)
  .parallel(y)
  .parallel(n)
  .parallel(co);

conv.compute_at(relu, xo)
  .vectorize(c, vec)
  .unroll(c)
  .unroll(x)
  .unroll(y)
  .update()
  .reorder(c, x, y, r.x, r.y, r.z, n)
  .vectorize(c, vec)
  .unroll(c)
  .unroll(x)
  .unroll(y)
  .unroll(r.x, 2);
```

We will consider only the case without parallelization to avoid the difference
in parallel runtimes generated by Halide and used by MLIR. This schedule
corresponds to a sequence of loop manipulations, unrolling and vectorization.
The following directives are present and can be mapped to transformations on
Linalg as described below.

*   `split` decomposes a loop dimension into two immediately nested loops with
    the inner loop having at most the given number of iterations. This can be
    understood as loop _strip-mining_ or a degenerate case of tiling a single
    dimension using any of `linalg.tile_` transform ops. We will be using
    `transform.structured.tile_using_forall` as this kind of loop is best
    supported by bufferization and can also be turned into a parallel loop later
    on. Unlike Halide, this doesn’t add new dimensions to the original
    operation, but rather creates a loop around it and rewrites the operation
    itself to operate on a subset of the original data.
*   `reorder` rearranges the loops arbitrarily. In Linalg representation, loops
    are implicit and are intended to remain so as long as possible to target
    microkernels. The order of implicit loops in a `linalg.generic` operation
    can be changed by using `transform.structured.interchange`, but this does
    not apply to named operations that need to be “generalized” first by calling
    `transform.structured.generalize`. However, this can only reorder implicit
    dimensions and not the explicit loops materialized by tiling operations that
    can no longer be “folded” into the original operation. Instead, we can
    leverage this behavior by materializing loops directly in the desired order
    by “tiling” to size 1.
*   `vectorize` indicates that the given dimension should be vectorized with the
    given factor; if the loop extent is larger than the factor, the loop is
    effectively split into two parts and the inner one is vectorized. On the
    contrary, structured Linalg op vectorization applies as a global
    transformation to all suitable operations at, e.g., a function scope via
    `transform.structured.vectorize_children_and_apply_patterns`. It relies on
    MLIR’s support for multidimensional vectors to directly map multidimensional
    tensors, which are later decomposed into operations on smaller
    hardware-compatible vectors during lowering.
*   `unroll` performs loop unrolling, fully or up to the given factor. It is
    equivalent to `transform.loop.unroll`.
*   `compute_at` indicates that the value of the function must be computed
    within the given loop that will be produced for another function; depending
    on the relation between loops surrounding functions, this corresponds to
    either a loop distribution or a producer/consumer fusion. Given that the
    Linalg representation starts in the fully distributed form, it can be
    represented as a sequence of `transform.structured.fuse_into_containing_op`
    that operates on `forall` loops materialized by tiling beforehand.


## Recreating the Loop Structure

The three first transformation directives for `relu` in the Halide schedule aim
at producing the following loop structure.

```python
for co
  for n
    for y
      for xo
        for xi
          for ci
            relu[n, y, xo*tile_h + xi, co*tile_w*vec + ci] = ...
```

Note that the outer part of the `c` gets hoisted from all of the surrounding
loops. The implicit loop order for the operation is `n, y, x, c`, so the `co`
loop needs to be materialized first in order to achieve the desired reordering.
The remaining dimensions can be materialized as loops in one transformation.

```mlir
    //                                                             [n  y  x  c]
    %co, %relu2 = transform.structured.tile_using_forall %relu
                                                        tile_sizes [0, 0, 0, 64]
    %n_y_xo, %relu3 = transform.structured.tile_using_forall %relu2
                                                        tile_sizes [1, 1, 5, 0]
```

This will result in the following loops being created in the IR with the nested
elementwise operation operating on a smaller subset of original data via
implicit loops.

```mlir
scf.forall (%co) in (2) {
  scf.forall (%n, %y, %xo) in (5, 80, 20) {
    tensor.extract_slice
    // Implicit dimensions [ni=0:1, y=0:1, xi=0:5, ci=0:64]
    %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> } // ...
    scf.forall.in_parallel {
      tensor.parallel_insert_slice // ...
    }
  }
}
```

The following loop restructuring transformations are `compute_at` and `reorder`
on the `conv` function that need to happen before loops are destroyed by
unrolling and vectorization. They intend to produce the final desired loop
structure.

```python
for co
  for n
    for y
      for xo
        for xi
          for ci
            conv[n, y, x*tile_h + xi, co*tile_w*vec + ci] = ...
        for rz
          for ry
            for rx
              for xi
                for ci
                  conv[n, y, x*tile_h + xi, co*tile_w*vec + ci] += ...
        for xi
          for ci
            relu[n, y, xo*tile_h + xi, co*tile_w*vec + ci] = ...
```

Practically, this corresponds to fusing the convolution initialization and
update into the `co, n, y, xo` loops materialized by tiling earlier. Structured
op transformation set supports fusing the producer of a value into its consumer,
so fusion happens in two stages:

*   first the main convolution update is fused into ReLU that uses it and has
    loops materialized;
*   then the bias initialization is fused into the convolution+relu loop nest.

Each stage consists of two transformations fusing the computational operation
into the outer loop, then the inner loop.

```mlir
%conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
%conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
  into %n_y_xo

%bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
%bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2
  into %n_y_xo2
```

To complete the structure, we need to put the `rz, ry, rx` loops outside the
“tile” loops `xi, ci`. This can be achieved materializing the corresponding
loops from the convolution operation. However, these are reduction loops and it
wouldn’t be valid to materialize them as intrinsically parallel “forall” loops.
Instead, we use the dedicated “reduction tiling” transformation and produce
sequential `scf.for` loops. (`scf.forall` loops can also express parallel
reductions, but the corresponding transformation doesn’t handle reductions along
more than one dimension at the moment of writing.)

```mlir
%rz_ry_rx, %red_fill, %conv4, %comb
  = transform.structured.tile_reduction_using_for %conv3
//               n  y  x  c  rz ry rx
  by tile_sizes=[0, 0, 0, 0, 1, 1, 1]
```

This transformation materializes the desired loops around the convolution
operation. It is also more capable than merely producing (reduction) loops: the
transformed code performs `tile_size` partial reductions of `N / tile_size`
elements, potentially in parallel by changing the dimension kind of the
structured operation inside the loop, and then performs a final reduction of
these partial results by producing a new “combiner” structured operation after
the loops. In our case, `tile_size = 1` along all dimensions, so the reduction
is entirely performed by the generated loops. The combiner structured operation
is still produced and adds up the reduction result with the initial value. This
changes the order of floating point operations (so would reduction tiling with
non-unit size) and may affect the final result due to non-commutativity of these
operations, but is explicitly allowed by `fastmath` flags. Halide also emits
LLVM IR with full `fastmath` flags.

Finally, we need to produce innermost loops `xi` and `ci` that are still not
explicit. As our next step is going to be vectorization along `ci`, we need to
take into account the way it operates on MLIR structured operations: rather than
selecting a specific vector size and loop/dimension to vectorize, it directly
substitutes multidimensional vector types for tensor types and updates the
operations accordingly. Therefore, our tensor type should not become trivial,
i.e. size-1, and retain a `vector_size` sized dimension along the desired axis,
`ci`. This can be achieved by tiling with `vector_size` as tile size in that
dimension:

```mlir
//                                                                  n  y  xi ci
%1, %c5 = transform.structured.tile_using_forall %conv4 tile_sizes [0, 0, 1, 16]
%2, %b4 = transform.structured.tile_using_forall %bias3 tile_sizes [0, 0, 1, 16]
%3, %r4 = transform.structured.tile_using_forall %relu3 tile_sizes [0, 0, 1, 16]
%4, %c2 = transform.structured.tile_using_forall %comb  tile_sizes [0, 0, 1, 16]
```

Note that the combiner operation produced by reduction tiling is also tiled here.


## Explicit Loop Unrolling

The remaining unhandled loop transformation is unrolling. Specifically,
unrolling is requested for the innermost loops that form the 4x5 tile of
16-element vector operations to ensure a contiguous sequence of `vfma`
instructions using 20 512-bit vector registers as accumulators. Unrolling
additional loops,, `unroll(y)` and `unroll(r.x, 2)`, is requested in the
schedule but _has no practical effect_. That is, the code, and all intermediate
representations, produced by Halide with these directives removed is _strictly
identical_ to the code with the full schedule. Therefore, we will only unroll
the corresponding loops corresponding to `xi` and `ci` dimensions that actually
get unrolled by Halide.

As tiling in the transform dialect produces handles to the loops materialized by
tiling, unrolling those loops is just a matter of chaining the corresponding
transformation. Note that the inner loop must be unrolled first as unrolling the
outer loop will invalidate the handles to the inner loop.

```mlir
transform.loop.unroll %bias_ci {factor = 4}
transform.loop.unroll %bias_xi {factor = 5}
transform.loop.unroll %conv_ci {factor = 4}
transform.loop.unroll %conv_xi {factor = 5}
transform.loop.unroll %relu_ci {factor = 4}
transform.loop.unroll %relu_xi {factor = 5}
transform.loop.unroll %comb_ci {factor = 4}
transform.loop.unroll %comb_xi {factor = 5}
```

## Vectorization

These transformations produced the desired loop structure and we are now ready
to vectorize. Before proceeding it is desirable to simplify the code as tiling
and fusion may have produced a lot of operations computing tensor subsets and
loop ranges, some of which may be duplicated or excessively complex.
Simplification involving canonicalization, common subexpression elimination,
loop invariant code motion and various rewrite patterns can be applied directly
from the transform dialect. Furthermore, an arbitrary combination of rewrite
patterns can be applied _in one sweep_ to a given scope, a functionality that
_cannot be achieved with conventional compiler passes_ that apply each group of
patterns separately (at least without creating a new pass for each combination
of pattern groups).

```mlir
%f00 = transform.structured.match ops{["func.func"]} in %arg0
transform.apply_patterns to %f00 {
  transform.apply_patterns.canonicalization
  transform.apply_patterns.linalg.tiling_canonicalization
}
transform.apply_cse to %f00

%all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0
transform.apply_licm to %all_loops
```

One final simplification is necessary to produce good vectorized code.
Tiling-by-one as a way of materializing loops produced structured (`linalg`)
operations processing 4D types where only one dimension isn’t unit-sized, e.g.,
`tensor<1x1x1x16xf32>` where 16 is the vector size corresponding to AVX512,
as structured tiling doesn’t modify the rank of the operation in order to
preserve the original structure. Even though the core computation is the same,
the produced code may end up more complicated than necessary, in particular when
decomposing multidimensional vectors into single-dimensional vectors supported
by hardware. Such unit dimensions can be explicitly folded away using the
corresponding pattern set before vectorization.

```mlir
transform.apply_patterns to %f00 {
  transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
}

%fv = transform.structured.vectorize_children_and_apply_patterns %f00
```

This produces the desired code performing arithmetic operations on
`vector<16xf32>` types that can be easily lowered to AVX512 instructions by the
downstream compiler. Vectorization may have created new opportunities for code
simplification, in particular combining tensor subsetting and vector slicing
operations. Another round of simplification can be applied post vectorization.

```mlir
transform.apply_patterns to %fv {
  transform.apply_patterns.canonicalization
  transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
}
transform.apply_cse to %fv
transform.structured.hoist_redundant_vector_transfers %fv
```

## Lowering to LLVM and The Bufferization Hurdle

With the loop restructuring done, the program now needs to be converted to the
executable form. The first step in doing so is _bufferization_, the process that
associates a memory buffer with every tensor in the payload IR. MLIR’s one-shot
bufferization is directly available as a transform operation.

```mlir
%arg1 = transform.bufferization.one_shot_bufferize %arg0 {
  bufferize_function_boundaries = true,
  function_boundary_type_conversion = 1 : i32 }
```

One-shot bufferization itself does not produce buffer deallocations, which may
lead to leaks. So we have to run the buffer deallocation pass pipeline to avoid
them. Note that the transform dialect seamlessly runs named passes and pass
pipelines: if desired, one could replace complex `--pass-pipeline expressions`
with operations. Note that we apply the pipeline to functions rather than entire
module to avoid running it on the transform IR that is contained in the module.

```mlir
%f = transform.structured.match ops{["func.func"]} in %arg1
  : (!transform.any_op) -> !transform.any_op
transform.apply_registered_pass "buffer-deallocation-pipeline" to %f
  : (!transform.any_op) -> !transform.any_op
```

In this particular case, the transformed IR could be directly bufferized. This
is not always the case in general as some operations, in particular
`tensor.empty` may not be bufferizable. Such operations need to be removed
before running the bufferization, which can often be achieved by sufficient
fusion (as in our case), or by running dedicated transformations
`transform.bufferization.eliminate_empty_tensors` that removes the
`tensor.empty` operations only serving for defining the size of a computation or
`transform.bufferization.empty_tensor_to_alloc_tensor` that materializes a new
temporary buffer for empty tensors to be used as local caches.

```mlir
// Apply general canonicalization and CSE to each function after
// bufferization as new simplification opportunities may have appeared.
%fb = transform.structured.match ops{["func.func"]} in %arg1
transform.apply_patterns to %fb {
  transform.apply_patterns.canonicalization
}
transform.apply_cse to %fb

// Lower complex, multidimensional vector operations into simpler
// primitives. This particular selection of the pattern groups corresponds
// to vector dialect operations present in the payload IR at this stage.
// Many of these groups can be parameterized to use different strategies or
// lower-level primitives offering performance trade-offs. In this case, we
// are selecting the simplest strategies.
transform.apply_patterns to %fb {
  transform.apply_patterns.vector.lower_contraction
    lowering_strategy = parallelarith
  transform.apply_patterns.vector.lower_transfer
    max_transfer_rank = 1
  transform.apply_patterns.vector.lower_transpose
    lowering_strategy = eltwise
  transform.apply_patterns.vector.lower_shape_cast
}

// These patterns apply in a separate sweep to avoid transfer-to-scf
// patterns overlap with lower-transfer patterns as they apply to the same
// kind of operations. These patterns may produce local allocations to act
// as temporary caches deep inside loops, which could lead to catastrophic
// performance. Such allocations are moved onto the stack and hoisted from
// all the surrounding loops.
transform.apply_patterns to %fb {
  transform.apply_patterns.vector.transfer_to_scf
  transform.apply_patterns.memref.alloc_to_alloca
  }
transform.bufferization.buffer_loop_hoisting %fb

// A final round of cleanups additionally includes patterns to simplify
// buffer aliasing operations that may have been introduced during
// bufferization and could result in excessively complex address
// computation.
transform.apply_patterns to %fb {
  transform.apply_patterns.memref.fold_memref_alias_ops
  transform.apply_patterns.canonicalization
}
transform.apply_cse to %fb
```

Due to its inter-procedural nature, one-bufferization processes the entire
payload module and thus invalidates all previously created handles. Therefore,
it is typically a late step in the transformation sequence where precise
targeting of transformation is no longer required. The following transformations
are typically module- or function-wide rewrites that are often pattern-based
lowerings. This part of the sequence can be seen as a pass pipeline specified
directly in the transform dialect, with pattern-based lowering passes
constructed _on-the-fly_ from named groups of patterns.

The resulting IR can be further completely lowered to the LLVM dialect, then to
LLVM IR and processed by the LLVM compiler to produce an executable or JITted.

The generated code runs in ~420ms on an Intel processor with Skylake
microarchitecture clocked at 2.0GHz. Given that the computation performs
$5*80*100*128*(2*3*3*128 + 2) ~= 5.9 * 10^9$ floating point operations, it
reaches ~14 GFlops. With 1 FMA unit available, the single-core performance of
the test processor is 64 GFlops $16 * 2 * 2 * 10^9$, where 16 is the vector
width), so only 22% of the theoretical peak is achieved.

The code produced by Halide runs in ~120ms on the same processor, a 3.5x
improvement and 77% of peak. Let us analyze the generated assembly to understand
the source of the difference. The main computational effort is expected to
happen around floating point multiplications and additions in the convolution.
In both cases, the assembly features AVX512 `vfma231ps` instructions operating
on `%zmm` 512-bit vector registers. In the MLIR-generated code, they are
interspersed with memory accesses loading _two _of the `fma` operands before
each operation and leading to increased latency.

```asm
vmovups       -192(%r10), %zmm0
vbroadcastss  -1536(%rdi,%r9), %zmm1
vmovups       112(%rsp), %zmm2
vfmadd231ps   %zmm1, %zmm0, %zmm2     # zmm2 = (zmm0 * zmm1) + zmm2
vmovups       %ymm2, 112(%rsp)
vextractf64x4 $1, %zmm2, 144(%rsp)
// 19 more blocks of either
//  (a) vmovups,vbroadcast,vfma(z,z),vextract,
//  (b) vbroadcast,vfma(z,mem),vextract
```

The Halide-generated code however features compact blocks of `vfma231ps` and
`vbroadcastss` loading one of the operands while the other two are resident in
registers and loaded before `fma`.

```asm
vbroadcastss    -1536(%rsi,%rbx), %zmm25
vmovups         -192(%rdi), %zmm26
vmovups         -128(%rdi), %zmm27
vmovups         -64(%rdi), %zmm28
vmovups         (%rdi), %zmm29
vfmadd231ps     %zmm25, %zmm26, %zmm24  # zmm24 = (zmm26 * zmm25) + zmm24
vfmadd231ps     %zmm25, %zmm27, %zmm23  # zmm23 = (zmm27 * zmm25) + zmm23
vfmadd231ps     %zmm25, %zmm28, %zmm22  # zmm22 = (zmm28 * zmm25) + zmm22
vfmadd231ps     %zmm25, %zmm29, %zmm21  # zmm21 = (zmm29 * zmm25) + zmm21
vbroadcastss    -1024(%rsi,%rbx), %zmm25
vfmadd231ps     %zmm25, %zmm26, %zmm20  # zmm20 = (zmm26 * zmm25) + zmm20
vfmadd231ps     %zmm25, %zmm27, %zmm19  # zmm19 = (zmm27 * zmm25) + zmm19
vfmadd231ps     %zmm25, %zmm28, %zmm18  # zmm18 = (zmm28 * zmm25) + zmm18
vfmadd231ps     %zmm25, %zmm29, %zmm17  # zmm17 = (zmm29 * zmm25) + zmm17
vbroadcastss    -512(%rsi,%rbx), %zmm25

// 3 more blocks of 4 vfmadd231 followed by a vbroadcast
```

Inspecting the progressive intermediate representations produced by MLIR, one
can observe the load(transfer)/fma interspersing at all levels starting after
schedule application. The repeated tensor subsetting operations, that are later
transformed into vector transfer operations, and vector memory loads, are
produced by loop unrolling that was explicitly requested in the schedule! The
issue is the single-assignment model of tensors (and vectors) that results in
long and complex chains of access and update operations that become so long that
the lower-level transformations and the downstream compiler can no longer
simplify them. In fact, unrolling loops early in the transformation sequence can
lead to all sorts of compiler-performance related problems (including the
compiler failing to perform some optimizations due to excessive code length) in
the process.

It is therefore desirable to perform loop unrolling at a later stage,
specifically after bufferization and relevant simplification. However,
bufferization invalidates all loop handles including to loops that we are
willing to unroll. This hurdle can be overcome by matching the payload IR
operations after bufferization to produce new handles. We will first change the
kind of loops produced in the schedule from `scf.for` to `scf.forall` to have
less operations to match by using `transform.structured.tile_using_forall`
instead of `transform.structured.tile` when tiling with sizes `[0, 0, 1, 16]`.
Then we can match all `scf.forall` operations in the payload IR and transform
them into single-iterator `scf.for` loops _after bufferization_.

```mlir
%foralls = transform.structured.match ops{["scf.forall"]} in %arg1
%xi_bias, %ci_bias = transform.loop.forall_to_for %xi_ci_bias
%xi_conv, %ci_conv = transform.loop.forall_to_for %xi_ci_conv
%xi_relu, %ci_relu = transform.loop.forall_to_for %xi_ci_relu
%xi_comb, %ci_comb = transform.loop.forall_to_for %xi_ci_comb
```

We can then move our loop unrolling transformations later in the transformation
sequence as desired. Compiling this new version to assembly produces exactly the
same core computation around `vfmadd231ps` as Halide’s version, which only
differs slightly in allocated registers. Unsurprisingly, this version runs
roughly in 120ms on the same machine.


## Multi-Dimensional Vectors to the Rescue

While we managed to produce similar code to Halide in the previous section, we
did so by rematching generated loops after bufferization, which partially defies
the purpose of using handles to chain transformations in the Transform dialect.
Luckily, this step is not really necessary. It only served as an exercise in
producing the desired loop structure.

Multidimensional structured operations on vectors are lowered to target-specific
vectors by unrolling and splitting. For example, an elementwise arithmetic
operation on `vector<5x64xf32>` is replaced with 5 operations on
`vector<64xf32>` and additional vector value manipulations to recreate the
required type at the MLIR level. Each of these operations is then split into 4
operations on `vector<16xf32>` at the LLVM level where the information about
the target vector width becomes available. Collectively, this has exactly the
same effect as first materializing the 5x4 loop nest, and then fully unrolling
these loops. Therefore, the last stage of tiling, re-matching and unrolling can
be removed from the schedule.

The resulting assembly has all `vbroadcast` grouped together before `vfmadd231`
but otherwise has a similar structure. This grouping is due to each
multi-dimensional vector operation being “unrolled” separately. When executed,
it runs in ~110ms, a slight improvement of 8% over both the previous version and
Halide, and reaches ~53.7 GFlop/s or 84% of peak single-core performance. The
improvement is largely due to the intermediate representation being shorter and
simpler in presence of large-vector operations, which allowed for more
aggressive address computation and load placement optimization.

The final transformation strategy is checked into the repository at
[mlir/examples/transform/ChH/full.mlir](
https://github.com/llvm/llvm-project/tree/main/mlir/test/Examples/transform/ChH/full.mlir).
