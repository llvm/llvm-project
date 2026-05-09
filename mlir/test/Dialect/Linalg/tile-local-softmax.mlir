// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// Test tiling local_softmax along the tn (tile number) dimension.
// This is the key tiling that enables fusion with the rescaling matmul.

// CHECK-LABEL: func.func @tile_local_softmax_tn
// CHECK-SAME:    %[[INPUT:.*]]: tensor<4x128xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:         %[[INPUT_OFFSET:.*]] = affine.apply
// CHECK:         %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[INPUT]][%[[C0]], %[[INPUT_OFFSET]]] [4, 32]
// CHECK:         %[[OUTPUT_SLICE:.*]] = tensor.extract_slice %{{.*}}[%[[C0]], %[[IV]], %[[C0]]] [4, 1, 32]
// CHECK:         %[[MAX_SLICE:.*]] = tensor.extract_slice %{{.*}}[%[[C0]], %[[IV]]] [4, 1]
// CHECK:         %[[DEN_SLICE:.*]] = tensor.extract_slice %{{.*}}[%[[C0]], %[[IV]]] [4, 1]
// CHECK:         linalg.local_softmax dimension(1) tile_size(32)
// CHECK-SAME:      ins(%[[INPUT_SLICE]] : tensor<4x32xf32>)
// CHECK-SAME:      outs(%[[OUTPUT_SLICE]] : tensor<4x1x32xf32>, %[[MAX_SLICE]] : tensor<4x1xf32>, %[[DEN_SLICE]] : tensor<4x1xf32>)
// CHECK:         tensor.insert_slice
// CHECK:         tensor.insert_slice
// CHECK:         tensor.insert_slice
// CHECK:         scf.yield
func.func @tile_local_softmax_tn(%input : tensor<4x128xf32>) -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %output = tensor.empty() : tensor<4x4x32xf32>
  %max = tensor.empty() : tensor<4x4xf32>
  %den = tensor.empty() : tensor<4x4xf32>
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.local_softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Tile along tn dimension (dim 1 of output) with tile_size=1.
    // This produces one tile per scf.for iteration.
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
