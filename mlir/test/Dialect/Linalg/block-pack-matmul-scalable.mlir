// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=[32],[16],[64] allow-padding=1" \
// RUN: -canonicalize -split-input-file | FileCheck %s --check-prefix=SCALABLE

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=[32],[16],[64] allow-padding=0" \
// RUN: -canonicalize -split-input-file | FileCheck %s --check-prefix=SCALABLE-NOPAD

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,[64] allow-padding=1" \
// RUN: -canonicalize -split-input-file | FileCheck %s --check-prefix=MIXED

// -----

// All-scalable block factors, static input shapes, allow-padding=1.
// Inner tile sizes and outer dimensions are all dynamic (vscale-relative).

func.func @block_matmul_scalable_static(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// SCALABLE-LABEL: func @block_matmul_scalable_static(
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[MB:.+]] = arith.muli %{{.+}}, %[[C32]] : index
// SCALABLE-DAG: %[[NB:.+]] = arith.muli %{{.+}}, %[[C16]] : index
// SCALABLE-DAG: %[[KB:.+]] = arith.muli %{{.+}}, %[[C64]] : index
// SCALABLE: linalg.pack %{{.+}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[KB]]]
// SCALABLE: linalg.pack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [%[[NB]], %[[KB]]]
// SCALABLE: linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[NB]]]
// SCALABLE: linalg.generic
// SCALABLE: linalg.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[NB]]]

// Scalable factors with allow-padding=0: transform does not apply.
// SCALABLE-NOPAD-LABEL: func @block_matmul_scalable_static(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.matmul
// SCALABLE-NOPAD-NOT: linalg.unpack

// -----

// All-scalable block factors, dynamic input shapes, allow-padding=1.
// Both outer tile counts and inner tile sizes are fully dynamic.

func.func @block_matmul_scalable_dynamic(
    %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// SCALABLE-LABEL: func @block_matmul_scalable_dynamic(
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[MB:.+]] = arith.muli %{{.+}}, %[[C32]] : index
// SCALABLE-DAG: %[[NB:.+]] = arith.muli %{{.+}}, %[[C16]] : index
// SCALABLE-DAG: %[[KB:.+]] = arith.muli %{{.+}}, %[[C64]] : index
// SCALABLE: linalg.pack %{{.+}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[KB]]]
// SCALABLE: linalg.pack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [%[[NB]], %[[KB]]]
// SCALABLE: linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[NB]]]
// SCALABLE: linalg.generic
// SCALABLE: linalg.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [%[[MB]], %[[NB]]]

// SCALABLE-NOPAD-LABEL: func @block_matmul_scalable_dynamic(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.matmul
// SCALABLE-NOPAD-NOT: linalg.unpack

// -----

// Mixed block factors (mb=32, nb=16 static; kb=[64] scalable), static input.
// Only the K-dimension inner tile is scalable; mb and nb remain static.

func.func @block_matmul_mixed_static(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// MIXED-LABEL: func @block_matmul_mixed_static(
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[KB:.+]] = arith.muli %{{.+}}, %[[C64]] : index
// MIXED: linalg.pack %{{.+}} outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, %[[KB]]]
// MIXED: linalg.pack %{{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, %[[KB]]]
// MIXED: linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// MIXED: linalg.generic
// MIXED: linalg.unpack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [32, 16]

