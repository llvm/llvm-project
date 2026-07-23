// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=[32],[16],[64] allow-padding=1" \
// RUN: -canonicalize -cse -split-input-file | FileCheck %s --check-prefix=SCALABLE

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=[32],[16],[64] allow-padding=0" \
// RUN: -canonicalize -cse -split-input-file | FileCheck %s --check-prefix=SCALABLE-NOPAD

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,[64] allow-padding=1" \
// RUN: -canonicalize -cse -split-input-file | FileCheck %s --check-prefix=MIXED

// M=128, N=128, K=128
func.func @block_matmul_static(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// SCALABLE-LABEL: func @block_matmul_static(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @block_matmul_static(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<128x128xf32>, tensor<128x128xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<128x128xf32>) -> tensor<128x128xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @block_matmul_static(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]

// M=?, N=?, K=?
func.func @matmul_dynamic(
    %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// SCALABLE-LABEL: func @matmul_dynamic(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @matmul_dynamic(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @matmul_dynamic(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]

// M=?, N=128, K=32
func.func @matmul_mixed_static_dynamic(
    %A: tensor<?x32xf32>, %B: tensor<32x128xf32>, %C: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x32xf32>, tensor<32x128xf32>)
                     outs(%C : tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}

// SCALABLE-LABEL: func @matmul_mixed_static_dynamic(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @matmul_mixed_static_dynamic(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x32xf32>, tensor<32x128xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<?x128xf32>) -> tensor<?x128xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @matmul_mixed_static_dynamic(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]

// B=4, M=128, N=128, K=128
func.func @batch_matmul_static(
    %A: tensor<4x128x128xf32>, %B: tensor<4x128x128xf32>, %C: tensor<4x128x128xf32>) -> tensor<4x128x128xf32> {
  %0 = linalg.batch_matmul ins(%A, %B : tensor<4x128x128xf32>, tensor<4x128x128xf32>)
                           outs(%C : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
  return %0 : tensor<4x128x128xf32>
}

// SCALABLE-LABEL: func @batch_matmul_static(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @batch_matmul_static(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<4x128x128xf32>, tensor<4x128x128xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @batch_matmul_static(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]

// B=?, M=?, N=?, K=?
func.func @batch_matmul_dynamic(
    %A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>, %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.batch_matmul ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                           outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// SCALABLE-LABEL: func @batch_matmul_dynamic(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @batch_matmul_dynamic(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @batch_matmul_dynamic(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]

// B=?, M=?, N=128, K=32
func.func @batch_matmul_mixed_static_dynamic(
    %A: tensor<?x?x32xf32>, %B: tensor<?x32x128xf32>, %C: tensor<?x?x128xf32>) -> tensor<?x?x128xf32> {
  %0 = linalg.batch_matmul ins(%A, %B : tensor<?x?x32xf32>, tensor<?x32x128xf32>)
                           outs(%C : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
  return %0 : tensor<?x?x128xf32>
}

// SCALABLE-LABEL: func @batch_matmul_mixed_static_dynamic(
// SCALABLE-DAG: %[[VS:.+]] = vector.vscale
// SCALABLE-DAG: %[[C32:.+]] = arith.constant 32 : index
// SCALABLE-DAG: %[[C16:.+]] = arith.constant 16 : index
// SCALABLE-DAG: %[[C64:.+]] = arith.constant 64 : index
// SCALABLE-DAG: %[[M_VS:.+]] = arith.muli %[[VS]], %[[C32]] : index
// SCALABLE-DAG: %[[N_VS:.+]] = arith.muli %[[VS]], %[[C16]] : index
// SCALABLE-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[N_VS]], %[[K_VS]]]
// SCALABLE: linalg.pack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]
// SCALABLE: linalg.unpack
// SCALABLE-SAME: inner_tiles = [%[[M_VS]], %[[N_VS]]]

// SCALABLE-NOPAD-LABEL: func @batch_matmul_mixed_static_dynamic(
// SCALABLE-NOPAD-NOT: linalg.pack
// SCALABLE-NOPAD: linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<?x?x32xf32>, tensor<?x32x128xf32>)
// SCALABLE-NOPAD-SAME: outs(%{{.*}} : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
// SCALABLE-NOPAD-NOT: linalg.unpack

// MIXED-LABEL: func @batch_matmul_mixed_static_dynamic(
// MIXED-DAG: %[[VS:.+]] = vector.vscale
// MIXED-DAG: %[[C64:.+]] = arith.constant 64 : index
// MIXED-DAG: %[[K_VS:.+]] = arith.muli %[[VS]], %[[C64]] : index
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [16, %[[K_VS]]]
// MIXED: linalg.pack
// MIXED-SAME: inner_tiles = [32, 16]
// MIXED: linalg.unpack
// MIXED-SAME: inner_tiles = [32, 16]
