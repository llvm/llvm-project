// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline="builtin.module(func.func(linalg-detensorize{aggressive-mode}))" | FileCheck %s

#map = affine_map<() -> ()>

func.func @detensor_simple(%arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> attributes {iree.module.export} {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%arg1, %arg2 : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: func @detensor_simple
// CHECK-SAME:    (%[[arg1:.*]]: tensor<f32>, %[[arg2:.*]]: tensor<f32>)
// CHECK-DAG:     %[[arg1_val:.*]] = tensor.extract %[[arg1]]
// CHECK-DAG:     %[[arg2_val:.*]] = tensor.extract %[[arg2]]
// CHECK:         %[[detensored_res:.*]] = arith.addf %[[arg1_val]], %[[arg2_val]]
// CHECK:         %[[new_tensor_res:.*]] = tensor.from_elements %[[detensored_res]]
// CHECK:         return %[[new_tensor_res]]

func.func @detensor_op_sequence(%arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> attributes {iree.module.export} {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%arg1, %arg2 : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<f32>

  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%arg1, %1 : tensor<f32>, tensor<f32>)
    outs(%3 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %5 = arith.mulf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>

  %6 = tensor.empty() : tensor<f32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%1, %4 : tensor<f32>, tensor<f32>)
    outs(%6 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %5 = arith.divf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>

  return %7: tensor<f32>
}
// CHECK-LABEL: func @detensor_op_sequence
// CHECK-SAME:    (%[[arg1:.*]]: tensor<f32>, %[[arg2:.*]]: tensor<f32>)
// CHECK-DAG:     %[[arg1_val:.*]] = tensor.extract %[[arg1]]
// CHECK-DAG:     %[[arg2_val:.*]] = tensor.extract %[[arg2]]
// CHECK:         %[[detensored_res:.*]] = arith.addf %[[arg1_val]], %[[arg2_val]]
// CHECK:         %[[detensored_res2:.*]] = arith.mulf %[[arg1_val]], %[[detensored_res]]
// CHECK:         %[[detensored_res3:.*]] = arith.divf %[[detensored_res]], %[[detensored_res2]]
// CHECK:         %[[new_tensor_res:.*]] = tensor.from_elements %[[detensored_res3]]
// CHECK:         return %[[new_tensor_res]]

func.func @detensor_multiple_ops(%arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> attributes {iree.module.export} {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%arg1, %arg2 : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg3, %arg4 : f32
    %3 = arith.mulf %2, %arg4 : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: func @detensor_multiple_ops
// CHECK-SAME:    (%[[arg1:.*]]: tensor<f32>, %[[arg2:.*]]: tensor<f32>)
// CHECK-DAG:     %[[arg1_val:.*]] = tensor.extract %[[arg1]]
// CHECK-DAG:     %[[arg2_val:.*]] = tensor.extract %[[arg2]]
// CHECK:         %[[detensored_res:.*]] = arith.addf %[[arg1_val]], %[[arg2_val]]
// CHECK:         %[[detensored_res2:.*]] = arith.mulf %[[detensored_res]], %[[arg2_val]]
// CHECK:         %[[new_tensor_res:.*]] = tensor.from_elements %[[detensored_res2]]
// CHECK:         return %[[new_tensor_res]]

func.func @detensor_foreign_op(%arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> attributes {iree.module.export} {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%arg1, %arg2 : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %2 = "foreign.do_something"(%arg3, %arg4) {} : (f32, f32) -> f32
    linalg.yield %2 : f32
  } -> tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: func @detensor_foreign_op
// CHECK-SAME:    (%[[arg1:.*]]: tensor<f32>, %[[arg2:.*]]: tensor<f32>)
// CHECK-DAG:     %[[arg1_val:.*]] = tensor.extract %[[arg1]]
// CHECK-DAG:     %[[arg2_val:.*]] = tensor.extract %[[arg2]]
// CHECK:         %[[detensored_res:.*]] = "foreign.do_something"(%[[arg1_val]], %[[arg2_val]])
// CHECK:         %[[new_tensor_res:.*]] = tensor.from_elements %[[detensored_res]]
// CHECK:         return %[[new_tensor_res]]
