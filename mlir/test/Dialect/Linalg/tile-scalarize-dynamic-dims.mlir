// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-scalarize-dynamic-dims" -scf-for-loop-canonicalization -canonicalize -split-input-file | \
// RUN:     FileCheck %s

// CHECK-LABEL: func @matmul_partly_dynamic_tensor(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x2000xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//       CHECK:   %[[UB1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//       CHECK:   %[[UB2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//       CHECK:   scf.for %[[IV0:.*]] = %[[C0]] to %[[UB1]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:.*]] = %[[C0]] to %[[UB2]] step %[[C1]]
//       CHECK:       %[[S1:.*]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1] : tensor<?x?xf32> to tensor<1x1xf32>
//       CHECK:       %[[S2:.*]] = tensor.extract_slice %[[ARG1]][%[[IV1]], 0] [1, 2000] [1, 1] : tensor<?x2000xf32> to tensor<1x2000xf32>
//       CHECK:       %[[S3:.*]] = tensor.extract_slice %{{.*}}[%[[IV0]], 0] [1, 2000] [1, 1] : tensor<?x2000xf32> to tensor<1x2000xf32>
//       CHECK:       linalg.matmul ins(%[[S1]], %[[S2]] : tensor<1x1xf32>, tensor<1x2000xf32>) outs(%[[S3]] : tensor<1x2000xf32>) -> tensor<1x2000xf32>
func.func @matmul_partly_dynamic_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x2000xf32>)
    -> tensor<?x2000xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %out = tensor.empty(%d0) : tensor<?x2000xf32>
  %r = linalg.matmul {__internal_linalg_transform__ = "tile"}
      ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x2000xf32>)
      outs(%out: tensor<?x2000xf32>) -> tensor<?x2000xf32>
  return %r : tensor<?x2000xf32>
}

// -----

// The input IR of this test case is a tiled and peeled linalg.matmul op.

// CHECK-LABEL: func @tiled_and_peeled_matmul(
//       CHECK:   linalg.matmul ins({{.*}} : tensor<32x259xf32>, tensor<259x258xf32>) outs({{.*}} : tensor<32x258xf32>) -> tensor<32x258xf32>
//       CHECK:   linalg.matmul ins({{.*}} : tensor<1x259xf32>, tensor<259x258xf32>) outs({{.*}} : tensor<1x258xf32>) -> tensor<1x258xf32>
#map0 = affine_map<(d0) -> (64, -d0 + 257)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 32) * 32)>
#map2 = affine_map<(d0)[s0] -> (d0 - (s0 floordiv 32) * 32)>

func.func @tiled_and_peeled_matmul(%arg0: tensor<257x259xf32>, %arg1: tensor<259x258xf32>, %arg2: tensor<257x258xf32>) -> tensor<257x258xf32> {
  %c257 = arith.constant 257 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<257x258xf32>) -> tensor<257x258xf32>
  %1 = scf.for %arg3 = %c0 to %c257 step %c64 iter_args(%arg4 = %0) -> (tensor<257x258xf32>) {
    %2 = affine.min #map0(%arg3)
    %3 = tensor.extract_slice %arg0[%arg3, 0] [%2, 259] [1, 1] : tensor<257x259xf32> to tensor<?x259xf32>
    %4 = tensor.extract_slice %arg4[%arg3, 0] [%2, 258] [1, 1] : tensor<257x258xf32> to tensor<?x258xf32>
    %5 = affine.apply #map1()[%2]
    %6 = scf.for %arg5 = %c0 to %5 step %c32 iter_args(%arg6 = %4) -> (tensor<?x258xf32>) {
      %10 = tensor.extract_slice %3[%arg5, 0] [32, 259] [1, 1] : tensor<?x259xf32> to tensor<32x259xf32>
      %11 = tensor.extract_slice %arg6[%arg5, 0] [32, 258] [1, 1] : tensor<?x258xf32> to tensor<32x258xf32>
      %12 = linalg.matmul {__internal_linalg_transform__ = "tile"} ins(%10, %arg1 : tensor<32x259xf32>, tensor<259x258xf32>) outs(%11 : tensor<32x258xf32>) -> tensor<32x258xf32>
      %13 = tensor.insert_slice %12 into %arg6[%arg5, 0] [32, 258] [1, 1] : tensor<32x258xf32> into tensor<?x258xf32>
      scf.yield %13 : tensor<?x258xf32>
    }
    %7 = arith.cmpi slt, %5, %2 : index
    %8 = scf.if %7 -> (tensor<?x258xf32>) {
      %10 = affine.apply #map2(%2)[%2]
      %11 = tensor.extract_slice %3[%5, 0] [%10, 259] [1, 1] : tensor<?x259xf32> to tensor<?x259xf32>
      %12 = tensor.extract_slice %6[%5, 0] [%10, 258] [1, 1] : tensor<?x258xf32> to tensor<?x258xf32>
      %13 = linalg.matmul {__internal_linalg_transform__ = "tile"} ins(%11, %arg1 : tensor<?x259xf32>, tensor<259x258xf32>) outs(%12 : tensor<?x258xf32>) -> tensor<?x258xf32>
      %14 = tensor.insert_slice %13 into %6[%5, 0] [%10, 258] [1, 1] : tensor<?x258xf32> into tensor<?x258xf32>
      scf.yield %14 : tensor<?x258xf32>
    } else {
      scf.yield %6 : tensor<?x258xf32>
    }
    %9 = tensor.insert_slice %8 into %arg4[%arg3, 0] [%2, 258] [1, 1] : tensor<?x258xf32> into tensor<257x258xf32>
    scf.yield %9 : tensor<257x258xf32>
  }
  return %1 : tensor<257x258xf32>
}
