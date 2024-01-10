// RUN: mlir-opt --transform-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @generalize_unary
func.func @generalize_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  // CHECK-NOT:   linalg.elemwise_unary
  //     CHECK:   linalg.generic
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @map_no_inputs(
func.func @map_no_inputs(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  // CHECK-NOT:   linalg.map
  //     CHECK:   linalg.generic
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
