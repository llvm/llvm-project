// RUN: mlir-opt -split-input-file -convert-tensor-to-linalg -cse -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// tensor.pad
//===----------------------------------------------------------------------===//
// CHECK-LABEL:   func @generalize_pad_tensor_static_shape(
// CHECK-SAME:                                             %[[IN:.*]]: tensor<1x28x28x1xf32>) -> tensor<1x32x32x1xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[INIT:.*]] = linalg.init_tensor [1, 32, 32, 1] : tensor<1x32x32x1xf32>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%[[C0]] : f32) outs(%[[INIT]] : tensor<1x32x32x1xf32>) -> tensor<1x32x32x1xf32>
// CHECK:           %[[PADDED:.*]] = tensor.insert_slice %[[IN]] into %[[FILL]][0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : tensor<1x28x28x1xf32> into tensor<1x32x32x1xf32>
// CHECK:           return %[[PADDED]] : tensor<1x32x32x1xf32>
func.func @generalize_pad_tensor_static_shape(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x32x32x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0]  {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x1xf32> to tensor<1x32x32x1xf32>
  return %0 : tensor<1x32x32x1xf32>
}

// CHECK-LABEL:   func @generalize_pad_tensor_dynamic_shape(
// CHECK-SAME:                                              %[[IN:.*]]: tensor<4x?x2x?xf32>,
// CHECK-SAME:                                              %[[OFFSET:.*]]: index) -> tensor<4x?x?x?xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<4x?x2x?xf32>
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[OUT_DIM2:.*]] = arith.addi %[[C2]], %[[OFFSET]] : index
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[DIM3:.*]] = tensor.dim %[[IN]], %[[C3]] : tensor<4x?x2x?xf32>
// CHECK:           %[[OUT_DIM3:.*]] = arith.addi %[[DIM3]], %[[OFFSET]] : index
// CHECK:           %[[INIT:.*]] = linalg.init_tensor [4, %[[DIM1]], %[[OUT_DIM2]], %[[OUT_DIM3]]] : tensor<4x?x?x?xf32>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<4x?x?x?xf32>) -> tensor<4x?x?x?xf32>
// CHECK:           %[[PADDED:.*]] = tensor.insert_slice %[[IN]] into %[[FILL]]{{\[}}%[[C0]], %[[C0]], %[[OFFSET]], %[[C0]]] [4, %[[DIM1]], 2, %[[DIM3]]] [1, 1, 1, 1] : tensor<4x?x2x?xf32> into tensor<4x?x?x?xf32>
// CHECK:           return %[[PADDED]] : tensor<4x?x?x?xf32>
// CHECK:         }
func.func @generalize_pad_tensor_dynamic_shape(%arg0: tensor<4x?x2x?xf32>, %arg1: index) -> tensor<4x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %out = tensor.pad %arg0 low[%c0, %c0, %arg1, %c0] high[%c0, %c0, %c0, %arg1]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index, %gen_arg4: index):
    tensor.yield %cst : f32
  } : tensor<4x?x2x?xf32> to tensor<4x?x?x?xf32>
  return %out : tensor<4x?x?x?xf32>
}
