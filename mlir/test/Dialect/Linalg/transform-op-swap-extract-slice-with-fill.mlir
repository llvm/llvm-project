// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

func.func @swap_fill_extract_slice(%init : tensor<?x?x?xf32>, %offset0: index, %size1: index) -> tensor<?x6xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%f0 : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = tensor.extract_slice %0[%offset0, 8, 4] [1, %size1, 6] [1, 3, 1]
    : tensor<?x?x?xf32> to tensor<?x6xf32>
  return %1: tensor<?x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.linalg.swap_extract_slice_with_fill
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @swap_fill_extract_slice
// CHECK: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EXT:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 8, 4] [1, %{{.*}}, 6] [1, 3, 1]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[F0]] : f32) outs(%[[EXT]] : tensor<?x6xf32>) -> tensor<?x6xf32>
// CHECK: return %[[FILL]] : tensor<?x6xf32>
