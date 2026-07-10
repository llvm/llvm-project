// RUN: mlir-opt -test-linalg-elementwise-fusion-patterns=fuse-multiuse-producer -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @multi_use_producer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>, %arg4 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%b0: f32, %b1 : f32, %b2 : f32):
    %1 = arith.addf %b0, %b1 : f32
    linalg.yield %1, %1 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0#1, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg4 : tensor<?x?xf32>) {
  ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
    %3 = arith.mulf %b0, %b1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %0#0, %0#1, %2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func @multi_use_producer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[RESULT:.+]]:3 = linalg.generic
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2

func.func @multi_use_producer_2(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  %1 = llvm.mlir.constant(31 : index) : i64
  %2 = tensor.empty() : tensor<1x32x32x8xf32>
  %3 = tensor.empty() : tensor<1x32x32x8xindex>
  %4:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } 
  ins(%arg0 : tensor<1x32x32x8xf32>) 
  outs(%2, %3 : tensor<1x32x32x8xf32>, tensor<1x32x32x8xindex>) {
    ^bb0(%in: f32, %out: f32, %out_0: index):
      %9 = linalg.index 1 : index
      linalg.yield %0, %9 : f32, index
  } -> (tensor<1x32x32x8xf32>, tensor<1x32x32x8xindex>)

  %5 = tensor.empty() : tensor<1x32x32x8xi64>
  %6:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } 
  ins(%arg0, %4#1 : tensor<1x32x32x8xf32>, tensor<1x32x32x8xindex>) 
  outs(%2, %5 : tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>) {
    ^bb0(%in: f32, %in_0: index, %out: f32, %out_1: i64):
      %9 = builtin.unrealized_conversion_cast %in_0 : index to i64
      linalg.yield %0, %9 : f32, i64
  } -> (tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>)

  %7 = tensor.empty() : tensor<1x32x32x8xi64>
  %8:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } 
  ins(%arg0, %4#1, %6#1 : tensor<1x32x32x8xf32>, tensor<1x32x32x8xindex>, tensor<1x32x32x8xi64>) 
  outs(%2, %7 : tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>) {
    ^bb0(%in: f32, %in_0: index, %in_1: i64, %out: f32, %out_2: i64):
      %9 = llvm.sub %1, %in_1 : i64
      linalg.yield %0, %9 : f32, i64
  } -> (tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>)
  return %8#0 : tensor<1x32x32x8xf32>
}
// CHECK-LABEL: func @multi_use_producer_2(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x32x32x8xf32>)
// CHECK-SAME: -> tensor<1x32x32x8xf32>
// CHECK: %[[C31:.+]] = llvm.mlir.constant(31 : index) : i64
// CHECK: %[[R0:.+]]:2 = linalg.generic {
// CHECK-SAME: ins(%[[ARG0]], %[[ARG0]], %[[ARG0]], %[[ARG0]], %[[ARG0]] : tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>)
// CHECK-SAME: outs(%[[INIT:.+]], %[[INIT_1:.+]] : tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>)
// CHECK: ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[IN_2:.+]]: f32, %[[IN_3:.+]]: f32, %[[IN_4:.+]]: f32, %[[OUT:.+]]: f32, %[[OUT_I:.+]]: i64):
// CHECK: %[[IDX_9:.+]] = linalg.index 1 : index
// CHECK: %[[C_9:.+]] = builtin.unrealized_conversion_cast %[[IDX_9]] : index to i64
// CHECK: %[[C_SUB:.+]] = llvm.sub %[[C31]], %[[C_9]] : i64
// CHECK: linalg.yield %[[C0:.+]], %[[C_SUB]] : f32, i64
// CHECK: } -> (tensor<1x32x32x8xf32>, tensor<1x32x32x8xi64>)
// CHECK: return %[[R0]]#0 : tensor<1x32x32x8xf32>
