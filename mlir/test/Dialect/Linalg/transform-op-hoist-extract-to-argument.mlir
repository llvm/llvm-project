// RUN: mlir-opt --transform-interpreter %s 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @hoist_1dim_simple(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ARG0]] : tensor<4xf32>)
// CHECK-SAME:    outs(%[[ARG1]] : tensor<4xf32>)
// CHECK:         ^bb0(%[[VAL_IN:.*]]: f32, %[[VAL_OUT:.*]]: f32):
// CHECK:           linalg.yield %[[VAL_IN]] : f32
func.func @hoist_1dim_simple(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%arg1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @hoist_1_keep_1(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x1x1290240xi32>, %[[ARG1:.*]]: tensor<1x1x1290240xf32>)
// CHECK:         %[[EMPTY:.*]] = tensor.empty()
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ARG0]] : tensor<1x1x1290240xi32>)
// CHECK-SAME:    outs(%[[EMPTY]] : tensor<1290240xi32>)
// CHECK:         ^bb0(%[[VAL_IN:.*]]: i32, %[[VAL_OUT:.*]]: i32):
// CHECK:           %[[ID:.*]] = linalg.index 0
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[VAL_IN]]
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract %[[ARG0]][%[[CAST]], {{.*}}, %[[ID]]]
// CHECK:           linalg.yield %[[EXTRACTED]] : i32
func.func @hoist_1_keep_1(%arg0: tensor<1x1x1290240xi32>, %arg1: tensor<1x1x1290240xf32>) -> tensor<1290240xi32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1290240xi32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%0 : tensor<1290240xi32>) {
  ^bb0(%out: i32):
    %id0 = linalg.index 0 : index
    %extracted = tensor.extract %arg0[%c0, %c0, %id0] : tensor<1x1x1290240xi32>
    %index = arith.index_cast %extracted : i32 to index
    %extracted_1 = tensor.extract %arg0[%index, %c0, %id0] : tensor<1x1x1290240xi32>
    linalg.yield %extracted_1 : i32
  } -> tensor<1290240xi32>
  return %1 : tensor<1290240xi32>
}

// CHECK-LABEL: func.func @hoist_combined_inputs(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x1x1x1290240xf32>)
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:    ins({{.*}}, {{.*}}, %[[ARG0]] : i32, tensor<i32>, tensor<1x1x1x1290240xf32>)
// CHECK:         ^bb0(%{{.*}}, %{{.*}}, %[[VAL_HOISTED:.*]]: f32, %{{.*}}):
// CHECK:           linalg.yield %[[VAL_HOISTED]] : f32
func.func @hoist_combined_inputs(%arg0: tensor<1x1x1x1290240xf32>) -> tensor<1290240xf32> {
  %c0 = arith.constant 0 : i32
  %ci_0 = arith.constant 0 : index
  %cd_0 = arith.constant dense<0> : tensor<i32>
  %output = tensor.empty() : tensor<1290240xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> ()> , affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
   ins(%c0, %cd_0 : i32, tensor<i32>)
   outs(%output : tensor<1290240xf32>) {
  ^bb0(%in: i32, %in1: i32, %out: f32):
    %not_dense = arith.index_cast %in : i32 to index
    %dense = arith.index_cast %in1 : i32 to index
    %id = linalg.index 0 : index
    %extracted = tensor.extract %arg0[%not_dense, %dense, %ci_0, %id] : tensor<1x1x1x1290240xf32>
    linalg.yield %extracted : f32
  } -> tensor<1290240xf32>
  return %13 : tensor<1290240xf32>
}

// CHECK-LABEL: func.func @hoist_all_constants(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[ARG0]] : tensor<4x4xf32>)
// CHECK:         ^bb0(%[[VAL_IN:.*]]: f32, %[[VAL_OUT:.*]]: f32):
// CHECK:           linalg.yield %[[VAL_IN]] : f32
func.func @hoist_all_constants(%input: tensor<4x4xf32>, %output: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  }
     outs(%output : tensor<4xf32>) {
    ^bb0(%out: f32):
    %extracted = tensor.extract %input[%c0, %c1] : tensor<4x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @no_hoist_function_arg_index(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4xf32>, %[[IDX:.*]]: index, %[[OUT:.*]]: tensor<4xf32>)
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-NOT:     ins(
// CHECK:         ^bb0(%[[VAL_OUT:.*]]: f32):
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract %[[ARG0]][%[[IDX]]]
// CHECK:           linalg.yield %[[EXTRACTED]] : f32
func.func @no_hoist_function_arg_index(%input: tensor<4xf32>, %idx: index, %output: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  }
     outs(%output : tensor<4xf32>) {
    ^bb0(%out: f32):
    %id0 = linalg.index 0 : index
    %extracted = tensor.extract %input[%idx] : tensor<4xf32>
    linalg.yield %extracted : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    
    %1 = transform.structured.hoist_extract_to_argument %0 
      : (!transform.any_op) -> !transform.any_op
    
    transform.yield
  }
}
