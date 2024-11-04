// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @eliminate_tensor_empty(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<50x91xf32>,
//   CHECK-NOT:   tensor.empty
//       CHECK:   %[[filled:.*]] = linalg.fill {{.*}} outs(%[[arg0]]
//       CHECK:   %[[matmul:.*]] = linalg.matmul {{.*}} outs(%[[filled]]
//       CHECK:   %[[generic:.*]] = linalg.generic {{.*}} outs(%[[matmul]]
//       CHECK:   return %[[generic]]
func.func @eliminate_tensor_empty(
    %arg0: tensor<50x91xf32>, %arg1: tensor<91xf32>, %arg2: tensor<50x1280xf32>,
    %arg3: tensor<1280x91xf32>) -> tensor<50x91xf32>
{
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<50x91xf32>
  %1 = linalg.fill ins(%cst : f32)
                    outs(%0 : tensor<50x91xf32>) -> tensor<50x91xf32>
  %2 = linalg.matmul
      ins(%arg2, %arg3 : tensor<50x1280xf32>, tensor<1280x91xf32>)
      outs(%1 : tensor<50x91xf32>) -> tensor<50x91xf32>
  %3 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %2 : tensor<91xf32>, tensor<50x91xf32>)
      outs(%arg0 : tensor<50x91xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %16 = arith.addf %in, %in_0 : f32
    linalg.yield %16 : f32
  } -> tensor<50x91xf32>
  return %3 : tensor<50x91xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.eliminate_empty_tensors %0 : !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    transform.yield
  }
}
