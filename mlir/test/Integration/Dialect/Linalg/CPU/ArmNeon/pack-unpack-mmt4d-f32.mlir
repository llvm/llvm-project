// REQUIRES: target={{(aarch64|arm64).*}}

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:   -cse -canonicalize  -test-lower-to-llvm \
// DEFINE:   -o %t

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon" \
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
/// HIGH-LEVEL OVERVIEW
///
/// End-to-end test for computing matrix-multiplication (f32, accumulating to
/// f32) using linalg.mmt4d. In particular, demonstrates how the following
/// MLIR sequence (implemented in @matmul_via_mmt4d):
///
///   A_pack = linalg.pack A
///   B_pack = linalg.pack B
///   C_pack = linalg.pack C
///   out_pack = linalg.mmt4d(A_pack, B_pack, C_pack)
///
/// is equivalent to:
///
///  linalg.matmul(A, B, C)
///
/// (implemented in @matmul).
///
/// NOTES ON IMPLEMENTATION
/// 1. linalg.matmul is only lowered via `-test-lower-to-llvm`, with no tiling or
///    vectorization - it's there to provide a reference check, not to
///    demonstrate a lowering path.
///
/// 2. Packing gives the tiles a statically-known shape (no masking in the
///    mmt4d loop) and makes them contiguous in memory.
///
/// 3. The MMT4D and Pack/Unpack Ops are kept in separate functions to isolate
///    the corresponding lowering and lowering configs.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// @main
//
// The main entry point that computes matrix multiplication via
// linalg.mmt4d and linalg.matmul. The output should be independent of the
// Linalg Op used.
//===----------------------------------------------------------------------===//
func.func @main() {
  %A = arith.constant dense<[
    [1.0, 3.0, 5.0],
    [2.0, 4.0, 6.0],
    [3.0, 5.0, 7.0],
    [4.0, 6.0, 8.0],
    [5.0, 7.0, 9.0]
  ]> : tensor<5x3xf32>
  %B = arith.constant dense<[
    [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0, 43.0],
    [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0, 41.0, 44.0],
    [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0]
  ]> : tensor<3x15xf32>
  %C = arith.constant dense<[
    [1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 61.0, 66.0, 71.0],
    [2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 67.0, 72.0],
    [3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 63.0, 68.0, 73.0],
    [4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 64.0, 69.0, 74.0],
    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
  ]> : tensor<5x15xf32>

  // VARIANT: Matrix multiplication via linalg.mmt4d
  // CHECK: Unranked Memref
  // CHECK:  [23,   55,   87,   119,   151,   183,   215,   247,   279,   311,   343,   375,   407,   439,   471]
  // CHECK:  [30,   71,   112,   153,   194,   235,   276,   317,   358,   399,   440,   481,   522,   563,   604]
  // CHECK:  [37,   87,   137,   187,   237,   287,   337,   387,   437,   487,   537,   587,   637,   687,   737]
  // CHECK:  [44,   103,   162,   221,   280,   339,   398,   457,   516,   575,   634,   693,   752,   811,   870]
  // CHECK:  [51,   119,   187,   255,   323,   391,   459,   527,   595,   663,   731,   799,   867,   935,   1003]
  %C_mmt4d = func.call @matmul_via_mmt4d(%A, %B, %C) : (tensor<5x3xf32>, tensor<3x15xf32>, tensor<5x15xf32>) -> tensor<5x15xf32>
  %C_mmt4d_cast = tensor.cast %C_mmt4d : tensor<5x15xf32> to tensor<*xf32>
  vector.print str "RESULT FROM linalg.mmt4d:\n"
  call @printMemrefF32(%C_mmt4d_cast) : (tensor<*xf32>) -> ()

  // VARIANT: Matrix multiplication via linalg.matmul (reference check)
  // CHECK: Unranked Memref
  // CHECK:  [23,   55,   87,   119,   151,   183,   215,   247,   279,   311,   343,   375,   407,   439,   471]
  // CHECK:  [30,   71,   112,   153,   194,   235,   276,   317,   358,   399,   440,   481,   522,   563,   604]
  // CHECK:  [37,   87,   137,   187,   237,   287,   337,   387,   437,   487,   537,   587,   637,   687,   737]
  // CHECK:  [44,   103,   162,   221,   280,   339,   398,   457,   516,   575,   634,   693,   752,   811,   870]
  // CHECK:  [51,   119,   187,   255,   323,   391,   459,   527,   595,   663,   731,   799,   867,   935,   1003]
  %C_matmul = func.call @matmul(%A, %B, %C) : (tensor<5x3xf32>, tensor<3x15xf32>, tensor<5x15xf32>) -> tensor<5x15xf32>
  %C_matmul_cast = tensor.cast %C_matmul : tensor<5x15xf32> to tensor<*xf32>
  vector.print str "RESULT FROM linalg.matmul:\n"
  call @printMemrefF32(%C_matmul_cast) : (tensor<*xf32>) -> ()

  return
}

//===----------------------------------------------------------------------===//
// @matmul
//
// Implements matrix-multiplication via linalg.matmul. Lowered only via
// `-test-lower-to-llvm` (no tiling, no vectorization): this is a reference
// check for @matmul_via_mmt4d, not a lowering path under test.
//===----------------------------------------------------------------------===//
func.func private @matmul(%A: tensor<5x3xf32>, %B: tensor<3x15xf32>, %C: tensor<5x15xf32>) -> tensor<5x15xf32> {
  %C_matmul = linalg.matmul ins(%A, %B: tensor<5x3xf32>, tensor<3x15xf32>)
                            outs(%C: tensor<5x15xf32>) -> tensor<5x15xf32>
  return %C_matmul : tensor<5x15xf32>
}

//===----------------------------------------------------------------------===//
// @pack_lhs
//
// Implements packing for the A matrix (LHS) in matrix multiplication. The
// inner tile size is fixed: 4 * 1.
//===----------------------------------------------------------------------===//
func.func private @pack_lhs(%A: tensor<5x3xf32>) -> tensor<2x3x4x1xf32> {
  %pad = arith.constant 0.0 : f32
  %A_pack_empty = tensor.empty() : tensor<2x3x4x1xf32>
  %A_pack = linalg.pack %A
    padding_value(%pad : f32)
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 1]
    into %A_pack_empty : tensor<5x3xf32> -> tensor<2x3x4x1xf32>
  return %A_pack : tensor<2x3x4x1xf32>
}

//===----------------------------------------------------------------------===//
// @pack_rhs
//
// Implements packing for the B matrix (RHS) in matrix multiplication. The
// inner tile size is fixed: 4 * 1.
//===----------------------------------------------------------------------===//
func.func private @pack_rhs(%B: tensor<3x15xf32>) -> tensor<4x3x4x1xf32> {
  %pad = arith.constant 0.0 : f32
  %B_pack_empty = tensor.empty() : tensor<4x3x4x1xf32>
  %B_pack = linalg.pack %B
    padding_value(%pad : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [4, 1]
    into %B_pack_empty : tensor<3x15xf32> -> tensor<4x3x4x1xf32>
  return %B_pack : tensor<4x3x4x1xf32>
}

//===----------------------------------------------------------------------===//
// @pack_acc
//
// Implements packing for the C matrix (accumulator) in matrix multiplication.
// The inner tile size is fixed: 4 * 4.
//===----------------------------------------------------------------------===//
func.func private @pack_acc(%C: tensor<5x15xf32>) -> tensor<2x4x4x4xf32> {
  %pad = arith.constant 0.0 : f32
  %C_pack_empty = tensor.empty() : tensor<2x4x4x4xf32>
  %C_pack = linalg.pack %C
    padding_value(%pad : f32)
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 4]
    into %C_pack_empty : tensor<5x15xf32> -> tensor<2x4x4x4xf32>
  return %C_pack : tensor<2x4x4x4xf32>
}

//===----------------------------------------------------------------------===//
// @unpack_acc
//
// Implements unpacking for the C matrix (accumulator) in matrix
// multiplication. The inner tile size is fixed: 4 * 4.
//===----------------------------------------------------------------------===//
func.func private @unpack_acc(%C_packed: tensor<2x4x4x4xf32>) -> tensor<5x15xf32> {
  %C_out_empty = tensor.empty() : tensor<5x15xf32>
  %C_out_unpack = linalg.unpack %C_packed
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [4, 4]
    into %C_out_empty : tensor<2x4x4x4xf32> -> tensor<5x15xf32>
  return %C_out_unpack: tensor<5x15xf32>
}

//===----------------------------------------------------------------------===//
// @matmul_via_mmt4d
//
// Implements matrix-multiplication via linalg.mmt4d.
//===----------------------------------------------------------------------===//
// CHECK-IR-LABEL: llvm.func @matmul_via_mmt4d
// CHECK-IR-COUNT-4: llvm.intr.fmuladd
func.func private @matmul_via_mmt4d(%A: tensor<5x3xf32>, %B: tensor<3x15xf32>, %C: tensor<5x15xf32>) -> tensor<5x15xf32> {
  // Pack input matrices
  %A_pack = func.call @pack_lhs(%A): (tensor<5x3xf32>) -> tensor<2x3x4x1xf32>
  %B_pack = func.call @pack_rhs(%B): (tensor<3x15xf32>) -> tensor<4x3x4x1xf32>
  %C_pack = func.call @pack_acc(%C): (tensor<5x15xf32>) -> tensor<2x4x4x4xf32>

  // MMT4D
  %mmt4d = linalg.mmt4d ins(%A_pack, %B_pack : tensor<2x3x4x1xf32>, tensor<4x3x4x1xf32>) outs(%C_pack : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>

  // Unpack the output
  %C_out_unpack = func.call @unpack_acc(%mmt4d) : (tensor<2x4x4x4xf32>) -> tensor<5x15xf32>
  return %C_out_unpack : tensor<5x15xf32>
}

//===----------------------------------------------------------------------===//
// TD Sequence
//===----------------------------------------------------------------------===//
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consumed}) {
    //==========================================================================
    // HANDLE MMT4D
    //==========================================================================
    %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)
    %mmt4d_func = transform.get_parent_op %mmt4d <isolated_from_above> : (!transform.any_op) -> !transform.op<"func.func">

    // Step 1: Tile
    // Tile the outer dims (m, n, k) one at a time; the inner tile
    // (m0, n0, k0) = (4, 4, 1) is kept whole.
    %tiled_mmt4d, %_:3 = transform.structured.tile_using_for %mmt4d tile_sizes [1, 1, 1, 0, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize
    transform.structured.vectorize %tiled_mmt4d create_named_contraction
      vector_sizes [1, 1, 1, 4, 4, 1] : !transform.any_op

    // Step 3: Simplify
    transform.apply_patterns to %mmt4d_func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">

    // Hoisting and LICM - not strictly required
    %mmt4d_func_h = transform.structured.hoist_redundant_vector_transfers %mmt4d_func
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %mmt4d_func_h
      : (!transform.op<"func.func">) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

    // Simplification
    transform.apply_patterns to %mmt4d_func_h {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // Lower to outer products: with k0 = 1, each `vector.contract` is a rank-1
    // update, i.e., one `fmla` per row of the 4x4 accumulator tile.
    transform.apply_patterns to %mmt4d_func_h {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    //==========================================================================
    // HANDLE PACK + UNPACK
    //==========================================================================
    %pack = transform.structured.match ops{["linalg.pack"]} in %module : (!transform.any_op) -> !transform.any_op
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1.1 Tile the linalg.pack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.1)
    %tiled_pack_op_p, %loops_pack:2 = transform.structured.tile_using_for %pack tile_sizes [1, 1]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 1.2 Tile the linalg.unpack Op so that we can decompose it into e.g. tensor.pad
    //    and other lower-level Ops (see step 2.2)
    %tiled_unpack_op_p, %loops_unpack:2 = transform.structured.tile_using_for %unpack tile_sizes [4, 4]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 2.1. Decompose tiled PackOp into lower-level Ops + simplify
    %func_op_pack = transform.get_parent_op %tiled_pack_op_p <isolated_from_above> : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.linalg.decompose_pad
    } : !transform.op<"func.func">
    transform.apply_patterns to %func_op_pack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 2.2. Decompose tiled UnpackOp into lower-level Ops + simplify
    %func_op_unpack = transform.get_parent_op %tiled_unpack_op_p <isolated_from_above> : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.linalg.decompose_pack_unpack
    } : !transform.op<"func.func">
    transform.apply_patterns to %func_op_unpack {
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    //==========================================================================
    // BUFFERIZATION
    //==========================================================================
    %bufferize = transform.bufferization.one_shot_bufferize %module
      <bufferize_function_boundaries = true> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  //==========================================================================
  // TD MATCHERS (helper hooks)
  //==========================================================================
  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

//===----------------------------------------------------------------------===//
// Function signatures
//===----------------------------------------------------------------------===//
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
