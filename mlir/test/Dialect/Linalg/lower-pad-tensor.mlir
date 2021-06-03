// RUN: mlir-opt -split-input-file --test-linalg-transform-patterns="test-transform-pad-tensor"  %s | FileCheck --check-prefix=CHECK %s

// CHECK-DAG:   #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2 + 1, d3 + 2)>
// CHECK-LABEL: func @pad_tensor_with_memrefs
func @pad_tensor_with_memrefs(%arg0: memref<1x28x28x1xf32>) -> memref<2x31x31x3xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = memref.tensor_load %arg0 : memref<1x28x28x1xf32>
  %1 = linalg.pad_tensor %0 low[1, 1, 1, 2] high[0, 2, 2, 0]  {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<1x28x28x1xf32> to tensor<2x31x31x3xf32>
  %2 = memref.buffer_cast %1 : memref<2x31x31x3xf32>
  return %2 : memref<2x31x31x3xf32>
}

// CHECK:       linalg.fill
// CHECK:       linalg.generic
// CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]]]

// -----

// CHECK-DAG:   #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0 + 1, d1 + 2, d2 + 2)>
// CHECK-LABEL: func @pad_tensor_no_memrefs
func @pad_tensor_no_memrefs(%arg0: tensor<1x28x28xf32>) -> tensor<2x32x32xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.pad_tensor %arg0 low[1, 2, 2] high[0, 2, 2]  {
  ^bb0(%arg1: index, %arg2: index, %arg3: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<1x28x28xf32> to tensor<2x32x32xf32>
  return %0 : tensor<2x32x32xf32>
}

// CHECK:       linalg.fill
// CHECK:       linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$MAP2]], #[[$MAP3]]]

// -----

// CHECK-DAG:   #[[$MAP4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 + 2, d2 + 2, d3)>
// CHECK-LABEL: func @pad_tensor_detailed
func @pad_tensor_detailed(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x32x32x1xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.pad_tensor %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0]  {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<1x28x28x1xf32> to tensor<1x32x32x1xf32>
  return %0 : tensor<1x32x32x1xf32>
}

// CHECK:      %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x28x28x1xf32>) -> tensor<1x32x32x1xf32>
// CHECK:      %[[CTE:.+]] = constant 0.000000e+00 : f32
// CHECK:      %[[TMP:.+]] = linalg.init_tensor [1, 32, 32, 1] : tensor<1x32x32x1xf32>
// CHECK:      %[[R1c:.+]] = linalg.fill
// CHECK:      %[[R2c:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[$MAP4]], #[[$MAP5]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK:        ins(%arg0 : tensor<1x28x28x1xf32>) outs(%1 : tensor<1x32x32x1xf32>)
// CHECK:      ^bb0(%[[VAL:.+]]: f32, %arg2: f32)
// CHECK:        linalg.yield %[[VAL]] : f32
// CHECK:      return %[[R2c:.+]]
