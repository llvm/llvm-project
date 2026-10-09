// REQUIRES: target={{(aarch64|arm64).*}}

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:   -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline \
// DEFINE:   -convert-bufferization-to-memref -cse -canonicalize -convert-vector-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-neon" -test-lower-to-llvm \
// DEFINE:   -o %t

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon" \
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
/// HIGH-LEVEL OVERVIEW
///
/// End-to-end test for `linalg.matmul` (f32, accumulating to f32) on dynamic
/// shapes, tiled and vectorized for NEON (implemented in @matmul). Unlike the
/// static shapes in pack-unpack-mmt4d-f32.mlir, the tile sizes here don't
/// evenly divide the matrix dimensions, so this is also the variant that
/// exercises masked vectorization (masks computed from runtime dimensions).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// @main
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

  // CHECK: Unranked Memref
  // CHECK:  [23,   55,   87,   119,   151,   183,   215,   247,   279,   311,   343,   375,   407,   439,   471]
  // CHECK:  [30,   71,   112,   153,   194,   235,   276,   317,   358,   399,   440,   481,   522,   563,   604]
  // CHECK:  [37,   87,   137,   187,   237,   287,   337,   387,   437,   487,   537,   587,   637,   687,   737]
  // CHECK:  [44,   103,   162,   221,   280,   339,   398,   457,   516,   575,   634,   693,   752,   811,   870]
  // CHECK:  [51,   119,   187,   255,   323,   391,   459,   527,   595,   663,   731,   799,   867,   935,   1003]
  %A_dyn = tensor.cast %A : tensor<5x3xf32> to tensor<?x?xf32>
  %B_dyn = tensor.cast %B : tensor<3x15xf32> to tensor<?x?xf32>
  %C_dyn = tensor.cast %C : tensor<5x15xf32> to tensor<?x?xf32>
  %C_matmul = func.call @matmul(%A_dyn, %B_dyn, %C_dyn) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %C_matmul_cast = tensor.cast %C_matmul : tensor<?x?xf32> to tensor<*xf32>
  vector.print str "RESULT FROM linalg.matmul:\n"
  call @printMemrefF32(%C_matmul_cast) : (tensor<*xf32>) -> ()

  return
}

//===----------------------------------------------------------------------===//
// @matmul
//
// Implements matrix-multiplication via linalg.matmul on dynamic shapes.
//===----------------------------------------------------------------------===//
func.func private @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %C_matmul = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %C_matmul : tensor<?x?xf32>
}

//===----------------------------------------------------------------------===//
// TD Sequence
//===----------------------------------------------------------------------===//
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
      : (!transform.any_op) -> !transform.any_op
    %func = transform.get_parent_op %matmul <isolated_from_above> : (!transform.any_op) -> !transform.op<"func.func">

    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [2, 4, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.structured.vectorize %tiled_matmul vector_sizes [2, 4, 1] : !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.sink_ops
    } : !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.op<"func.func">

    transform.yield
  }
}

//===----------------------------------------------------------------------===//
// Function signatures
//===----------------------------------------------------------------------===//
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
