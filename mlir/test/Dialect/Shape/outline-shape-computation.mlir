// RUN: mlir-opt -allow-unregistered-dialect -outline-shape-computation %s | FileCheck %s


func.func @concat(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.dim %arg0, %c2 : tensor<?x4x?xf32>
  %1 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  %2 = tensor.dim %arg0, %c0 : tensor<?x4x?xf32>
  %3 = arith.addi %2, %c2 : index
  %4 = shape.from_extents %3, %c4, %0 : index, index, index
  %5 = shape.with_shape %1, %4 : tensor<?x4x?xf32>, !shape.shape
  return %1 : tensor<?x4x?xf32>
}

// CHECK: func.func @concat(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32, #shape.ext_info<symbols = [@shape_cal_0, @arg_0_dim_0, @arg_0_dim_2]>> {
// CHECK-NEXT:     %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32, #shape.ext_info<symbols = [@shape_cal_0, @arg_0_dim_0, @arg_0_dim_2]>>
// CHECK-NEXT:     return %0 : tensor<?x4x?xf32, #shape.ext_info<symbols = [@shape_cal_0, @arg_0_dim_0, @arg_0_dim_2]>>

// CHECK: shape.func private @shape_cal_0(%arg0: index) -> index {
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %0 = arith.addi %arg0, %c2 : index
// CHECK-NEXT:     return %0 : index
