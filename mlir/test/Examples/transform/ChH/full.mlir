// RUN: mlir-opt %s --test-transform-dialect-interpreter \
// RUN:             --test-transform-dialect-erase-schedule \
// RUN:             --math-uplift-to-fma \
// RUN:             --test-lower-to-llvm |\
// RUN: FileCheck %s

// Fixed-size tensor types to be used in convolution.
// Named sizes are: N=5 OH=80 OW=100 F=C=128 KH=KW=3.
// Input is NHWC.
// Filter is CHWF.
// Ouptut is NHWF.
!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

// Function containing the convolution. Note that its arguments and results are
// tensors annotated with attributes from the `bufferization` dialect. These
// attributes hint the bufferization pass to assume buffers can be directly
// used for these tensors without reshaping.
func.func @conv(
    %input: !tinput {bufferization.writable = false,
                     bufferization.access = "read",
                     bufferization.buffer_layout =
                         affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
    %filter: !tfilter {bufferization.writable = false,
                      bufferization.access = "read",
                      bufferization.buffer_layout =
                          affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
    %bias: !tbias {bufferization.writable = false,
                   bufferization.access = "read",
                   bufferization.buffer_layout = affine_map<(d0)->(d0)>},
    %output: !toutput {bufferization.writable = true,
                       bufferization.buffer_layout =
                           affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       bufferization.access = "write"}) -> !toutput
  // This requests a C-compatible interface to be emitted for the function
  // when translating to LLVM IR.
  attributes { llvm.emit_c_interface }
{
  // Bias. Using a named Linalg operation for brevity.
  %bias_init = tensor.empty() : !toutput
  %biased = linalg.broadcast ins(%bias : !tbias)
    outs(%bias_init : !toutput) dimensions = [0, 1, 2]

  // Convolution proper. While Linalg has named operations for 2D convolutions,
  // the one in the Halide example has an uncommon order of filter dimensions
  // and is not supported. It also takes the fitler as first argument. This
  // code recreates it faithfully using the generic form.
  %convolved = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction", "reduction", "reduction"],
    indexing_maps = [
      affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
    ]
  } ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput) {
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

  return %relued : !toutput
}

// Module containing the transformation script to be applied. The attribute
// is required to correctly verify the use of named (macro-like) sequences.
module attributes { transform.with_named_sequence } {
  // Apply transformations in a sequence to recreate the following Halide
  // schedule:
  //
  //   Var co, ci, xo, xi;
  //   relu.split(c, co, ci, vec * tile_w)
  //       .split(x, xo, xi, tile_h)
  //       .reorder(ci, xi, xo, y, n, co)
  //       .vectorize(ci, vec)
  //       .unroll(ci)
  //       .unroll(xi);
  //   conv.compute_at(relu, xo)
  //       .vectorize(c, vec)
  //       .unroll(c)
  //       .unroll(x)
  //       .unroll(y)
  //       .update()
  //       .reorder(c, x, y, r.x, r.y, r.z, n)
  //       .vectorize(c, vec)
  //       .unroll(c)
  //       .unroll(x)
  //       .unroll(y)
  //       .unroll(r.x, 2);
  //
  // where tile_w = 4, tile_h = 5, vec = 16. Note that unroll(y) and unroll(r.x)
  // have no effect on the Halide IR as of 294f80c49bf3bb8582446613c25fcce03b82.
  // Also note that the order of dimensions in Halide is inverted, e.g., co and
  // n are the outermost loops in the respective reorder directives.
  transform.sequence failures(propagate) {
  // This argument will point to the top-level module.
  ^bb0(%arg0: !transform.any_op):

    // 1. Find the operations we are going to transform usnig their names. This
    // is a simplistic approach that works when there are few operations in the
    // IR to be transformed. More complex scenarios should rely on operations
    // with `transform.match` prefix that are out of scope for this chapter.
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %conv, %relu = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2. Initial tiling to start producing the loop structure. Note that the
    // linalg.generic operation has the implicit loop order (n, y, x, c). Since
    // the desired order of dimensions is (co, n, y, xo, xi, ci), we first tile
    // only the c dimension to materialize the outermost co loop, and then tile
    // the other dimensions since they are already in the expected order. Tiling
    // by 1 produces the loop that iterates along the entire dimension. Tiling
    // by 0 does not produce a loop. The size 64 is chosen as tiling by 4*16
    // where 16 is the AVX512 vector length. Note that structured tiling doesn't
    // remove the dimensions that became trivial (unit size) so the resulting
    // sturucture is technically (co, no=n, yo=y, xo, [ni=1, yi=1, xi, ci])
    // where brackets indicate implicit loops of the `linalg.generic` operation
    // inside the loops produced by tiling.
    //
    //                                                             [n  y  x  c]
    %relu2, %co = transform.structured.tile_using_forall %relu
                                                        tile_sizes [0, 0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2
                                                        tile_sizes [1, 1, 5, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Compute_at is actually fusion into the given loop (given that we start
    // with totally fissioned form, Halide starts with a fused form by reusing
    // the loop iterators).
    %conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
      into %n_y_xo
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    // Also fuse the bias that we represent as a separate operation and Halide
    // represents as the "pure" (as opposed to "update") part of the conv
    // expression. Note that fusion consumes both handles and produces new
    // handles for chaining purposes.
    %bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2
      into %n_y_xo2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    // Clean up the result of fusion, which mechanically duplicates the producer
    // operation in the consumer loop without removing the original operation.
    // The original operation is now "dead": it has no uses and no side effects
    // so it can be removed by dead-code elimination (DCE) that runs as part of
    // pattern rewriting. The transform dialect allows to apply a combination
    // of named pattern sets, exposed as operations, in one sweep to an
    // isolated-from-above container payload operation. Note that we don't
    // actually need any patterns for DCE to run, just trigger the rewriting.
    //
    // This step is optional. The transformation can continue without it and
    // produce the same final IR, but makes it easier to manually examine the
    // intermediate stages.
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op

    // The loop reordering requested for the convolution operation requires
    // putting reduction loops (r.z, r.y. r.x) before the "inner" loops xi, ci.
    // The "inner" loops are still implicit as part of the linalg.generic
    // operation, and we need to materialize reduction loops around it by tiling
    // with size 1. Since we are producing reduction loops, we indicate that we
    // are tiling a reduction and request a sequential `scf.for` loops (parallel
    // reductions are supported by `scf.forall`, but we don't need those here).
    //
    // This transform operation is more capable than merely producing
    // (reduction) loops: the transformed code performs `tile_size` partial
    // reductions of `N / tile_size` elements, potentially in parallel by
    // changing the dimension kind of the structured operation inside the loop,
    // and then performs a final reduction of these partial results by producing
    // a new “combiner” structured operation after the loops. In our case,
    // tile_size = 1 along all dimensions, so the reduction is entirely
    // performed by the generated loops. The combiner structured operation is
    // still produced and adds up the reduction result with the initial value.
    %red_fill, %conv4, %combining, %rz_ry_rx
    = transform.structured.tile_reduction_using_for %conv3 by
    //            n  y  x  c  rz ry rx
      tile_sizes=[0, 0, 0, 0, 1, 1, 1]
      : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)

    // At this point, the inner Linalg operations have implicit iteration spaces
    // of 5x64 size, with some additional unit-size dimensions. Completely
    // replicating Halide schedule would require materializing the loops with
    // 5 and 4 iterations, respectively, unrolling those loops and marking the
    // remaining 16-point iteration space for vectorization.
    //
    // This is unnecessary in MLIR that supports multi-dimensional vectors,
    // which will be decomposed into target-specific sizes during the lowering.
    // Therefore, this schedule stops here.

    // Transform the named broadcast operation used for bias into the generic
    // form before vectorization to prevent special cases from kicking in.
    transform.structured.generalize %bias3
      : (!transform.any_op) -> !transform.any_op

    // Use the named macro to perform most of the lowering.
    transform.include @lower failures(propagate) (%arg0)
      : (!transform.any_op) -> ()
    transform.yield
  }

  // Named sequence of transformations is a macro-like object that can be
  // included from another place in the transform dialect, but doesn't allow for
  // recursion. This can be reused in other scenarios.
  transform.named_sequence @lower(
      %arg0: !transform.any_op {transform.consumed}) {
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Simplify the code as tiling and fusion may have produced a lot of
    // operations computing tensor subsets and loop ranges, some of which may be
    // duplicated or excessively complex. Simplification involving
    // canonicalization, common subexpression elimination, loop invariant code
    // motion and various rewrite patterns can be applied directly from the
    // transform dialect. Furthermore, an arbitrary combination of rewrite
    // patterns can be applied in one sweep to a given scope, a functionality
    // that cannot be achieved with conventional compiler passes that apply each
    // group of patterns separately (at least without creating a new pass for
    // each combination of pattern groups).
    transform.apply_patterns to %f00 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %f00 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
      in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op

    // Tiling-by-one as a way of materializing loops produced operations
    // processing 4+D types where only a handful of dimension isn’t unit-sized,
    // e.g., tensor<1x1x1x5x64xf32> where 5 and 64 are tile sizes. Remove such
    // unit dimensions before vectorization, for clarity.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    // Vectorize the remaining non-unit dimensions in structured operations.
    // This essentially rewrites operations on `tensor<5x64xf32>` into
    // opreations on `vector<5x64xf32>`. Further lowering in MLIR and LLVM will
    // decompose this into a sequence of operations on single-dimensional
    // vectors of the platform-relevant size, e.g., `vector<16xf32>` for AVX512.
    // High-level vector primitives, such as `vector.transpose` and
    // `vector.broadcast` can be introduced at this stage. They will be later
    // lowered to sequences of lower-level primitives such as `vector.shuffle`
    // depending on the selected lowering strategy.
    %fv = transform.structured.vectorize_children_and_apply_patterns %f00
      : (!transform.any_op) -> !transform.any_op

    // Vectorization may have created new opportunities for cleanups. In
    // particular, tensor subsetting operations can be composed with vector
    // operations, and vector transfer (multi-dimensional load/store) operations
    // can be recombined and hoisted out of loops.
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %fv
      : (!transform.any_op) -> !transform.any_op

    // Apply bufferization that rewrites the remaining operations on tensors
    // as operations on structured buffer (memref) types, including the function
    // API. MLIR bufferization uses destination-passing style meaning that a
    // buffer is shared between one of the operation's operands and its result.
    //
    // Since bufferization rewrites function signatures, it is applied as a
    // module-wise transformation. Therefore, it invalidates all previously
    // defined handles. Bufferization is usually a late step in the
    // transformation process, so invalidation is not an issue. However, if
    // other transformations, such as loop unrolling, are required after
    // bufferization, new handles should be produced using the match operations.
    %arg1 = transform.bufferization.one_shot_bufferize %arg0 {
      bufferize_function_boundaries = true,
      function_boundary_type_conversion = 1 : i32 }
      : (!transform.any_op) -> !transform.any_op

    // Apply general canonicalization and CSE to each function after
    // bufferization as new simplification opportunities may have appeared.
    %fb = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

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
    } : !transform.any_op

    // These patterns apply in a separate sweep to avoid transfer-to-scf
    // patterns overlap with lower-transfer patterns as they apply to the same
    // kind of operations. These patterns may produce local allocations to act
    // as temporary caches deep inside loops, which could lead to catastrophic
    // performance. Such allocations are moved onto the stack and hoisted from
    // all the surrounding loops.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.transfer_to_scf
      transform.apply_patterns.memref.alloc_to_alloca
      } : !transform.any_op
    transform.bufferization.buffer_loop_hoisting %fb : !transform.any_op

    // A final round of cleanups additionally includes patterns to simplify
    // buffer aliasing operations that may have been introduced during
    // bufferization and could result in excessively complex address
    // computation.
    transform.apply_patterns to %fb {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    transform.yield
  }
}

// The core computation, at the LLVM dialect level, must correspond to five
// immediately adjacent fma on vector<64xf32>.

// CHECK:      %[[R0:.+]] = llvm.mlir.undef : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[LINE0:.+]] = llvm.extractvalue %[[V:.+]][0] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA0:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE0]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R1:.+]] = llvm.insertvalue %[[FMA0]], %[[R0]][0]

// CHECK-NEXT: %[[LINE1:.+]] = llvm.extractvalue %[[V:.+]][1] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA1:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE1]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R2:.+]] = llvm.insertvalue %[[FMA1]], %[[R1]][1]

// CHECK-NEXT: %[[LINE2:.+]] = llvm.extractvalue %[[V:.+]][2] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA2:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE2]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R3:.+]] = llvm.insertvalue %[[FMA2]], %[[R2]][2]

// CHECK-NEXT: %[[LINE3:.+]] = llvm.extractvalue %[[V:.+]][3] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA3:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE3]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R4:.+]] = llvm.insertvalue %[[FMA3]], %[[R3]][3]

// CHECK-NEXT: %[[LINE4:.+]] = llvm.extractvalue %[[V:.+]][4] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA4:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE4]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R5:.+]] = llvm.insertvalue %[[FMA4]], %[[R4]][4]
