// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Test case: Most basic case. Adding a vector to itself.

#map = affine_map<(d0) -> (d0)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @basic
func.func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP]], #[[$MAP]]]
  // CHECK:   attrs =  {someattr}
  // CHECK:   ^bb0(%[[BBARG:.*]]: f32, %{{.*}}: f32):
  // CHECK:     arith.addf %[[BBARG]], %[[BBARG]]
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
     ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg0 : tensor<?xf32>) attrs = {someattr} {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Test case: Different indexing maps mean that args are not redundant, despite
// being the same Value.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @distinct_affine_maps
func.func @distinct_affine_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
     ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test case: Check rewriting mechanics for mixed redundant and
// non-redundant args.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @mixed_redundant_non_redundant
func.func @mixed_redundant_non_redundant(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
  // CHECK:   ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{[a-zA-Z0-9]+}}: f32):
  // CHECK:     "test.elementwise_mappable"(%[[BBARG0]], %[[BBARG1]], %[[BBARG0]])
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
     ins(%arg0, %arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %1 = "test.elementwise_mappable"(%arg1, %arg2, %arg3) : (f32, f32, f32) -> f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test case: Check rewriting mechanics for multiple different redundant args.

#map = affine_map<(d0) -> (d0)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @multiple_different_redundant_args
func.func @multiple_different_redundant_args(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic{{.*}}[#[[$MAP]], #[[$MAP]], #[[$MAP]]]
  // CHECK:   ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{[a-zA-Z0-9]+}}: f32):
  // CHECK:     "test.elementwise_mappable"(%[[BBARG0]], %[[BBARG1]], %[[BBARG0]], %[[BBARG1]])
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
     ins(%arg0, %arg1, %arg0, %arg1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
    outs(%arg0 : tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
    %1 = "test.elementwise_mappable"(%arg2, %arg3, %arg4, %arg5) : (f32, f32, f32, f32) -> f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Drop dead result.

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
func.func @drop_dead_results(%arg0 : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %0:4 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<?x?x?xf32>)
      outs(%arg0, %arg0, %arg0, %arg0
          : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32, %b4 : f32) :
      %1 = arith.addf %b0, %b0: f32
      linalg.yield %1, %1, %1, %1 : f32, f32, f32, f32
    } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %0#0, %0#2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>     
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
//      CHECK: func @drop_dead_results(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>)
//      CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:       outs(%[[ARG0]], %[[ARG0]] :
//      CHECK:   return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Current argmax lowering to `linalg.generic`. Cannot drop the
// first return even though it isnt used since it has an internal
// use.
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @argmax_lowering(%arg0 : tensor<?xf32>) -> tensor<i32> {
  %init0 = tensor.empty() : tensor<f32>
  %init1 = tensor.empty() : tensor<i32>
  %0:2 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["reduction"]}
    ins(%arg0 : tensor<?xf32>)
    outs(%init0, %init1 : tensor<f32>, tensor<i32>) {
  ^bb0(%b0: f32, %b1: f32, %b2: i32):
    %8 = linalg.index 0 : index
    %9 = arith.index_cast %8 : index to i32
    %10 = arith.cmpf oge, %b0, %b1 : f32
    %11 = arith.select %10, %b0, %b1 : f32
    %12 = arith.cmpf oeq, %b0, %b1 : f32
    %13 = arith.minsi %9, %b2 : i32
    %14 = arith.select %10, %9, %b2 : i32
    %15 = arith.select %12, %13, %14 : i32
    linalg.yield %11, %15 : f32, i32
  } -> (tensor<f32>, tensor<i32>)
  return %0#1 : tensor<i32>
}
//      CHECK: func @argmax_lowering(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?xf32>
//  CHECK-DAG:   %[[INIT0:.+]] = tensor.empty() : tensor<f32>
//  CHECK-DAG:   %[[INIT1:.+]] = tensor.empty() : tensor<i32>
//      CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME:       outs(%[[INIT0]], %[[INIT1]] :
//      CHECK:   return %[[GENERIC]]#1

// -----

// Do not remove operand needed for loop dim.
func.func @loop_dim_operand(%arg0 : tensor<?xf32>) -> tensor<i32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<i32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<i32>) -> tensor<i32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%arg0 : tensor<?xf32>) outs(%fill : tensor<i32>) {
    ^bb0(%b0: f32, %b1: i32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.addi %b1, %2 : i32
      linalg.yield %3 : i32
    } -> tensor<i32>
  return %0 : tensor<i32>
}
//      CHECK: func @loop_dim_operand(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:       ins(%[[ARG0]] :

// -----

// Do not remove outs operand needed for loop bound computation.
func.func @loop_dim_outs_operand(%arg0 : index) -> tensor<i32> {
  %cst = arith.constant 0 : i32
  %init1 = tensor.empty(%arg0) : tensor<?xi32>
  %init = tensor.empty() : tensor<i32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<i32>) -> tensor<i32>
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["parallel"]}
      outs(%init1, %fill : tensor<?xi32>, tensor<i32>) {
    ^bb0(%b0: i32, %b1: i32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.addi %b1, %2 : i32
      linalg.yield %2, %3 : i32, i32
    } -> (tensor<?xi32>, tensor<i32>)
  return %0#1 : tensor<i32>
}
//      CHECK: func @loop_dim_outs_operand(
// CHECK-SAME:     %[[ARG0:.+]]: index
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[ARG0]])
//      CHECK:   linalg.generic
// CHECK-SAME:       outs(%[[INIT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
func.func @multiple_redundant_args(%arg0 : tensor<?x?xi32>, %arg1 : tensor<?xi32>,
    %arg2 : tensor<?xi32>, %arg3 : tensor<?x?xi32>, %arg4 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.generic {
      indexing_maps = [#map3, #map0, #map0, #map2, #map1, #map1, #map2],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg4, %arg0, %arg0, %arg1, %arg3, %arg3
          : tensor<?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%arg2 : tensor<?xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32, %b4 : i32, %b5 : i32, %b6 : i32):
      %1 = arith.addi %b0, %b1 : i32
      %2 = arith.addi %1, %b2 : i32
      %3 = arith.addi %2, %b3 : i32
      %4 = arith.addi %3, %b4 : i32
      %5 = arith.addi %4, %b5 : i32
      %6 = arith.addi %5, %b6 : i32
      linalg.yield %6 : i32
    } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @multiple_redundant_args(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?xi32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]]: tensor<?xi32>)
//      CHECK:   %[[RETURN:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP2]]]
// CHECK-SAME:       iterator_types = ["parallel", "reduction"]
// CHECK-SAME:       ins(%[[ARG4]], %[[ARG0]], %[[ARG1]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[ARG2]] :
//      CHECK:   ^{{.+}}(%[[B0:[a-zA-Z0-9]+]]: i32
// CHECK-SAME:       %[[B1:[a-zA-Z0-9_]+]]: i32
// CHECK-SAME:       %[[B2:[a-zA-Z0-9_]+]]: i32
// CHECK-SAME:       %[[B3:[a-zA-Z0-9_]+]]: i32
// CHECK-SAME:       %[[B4:[a-zA-Z0-9_]+]]: i32)
//      CHECK:     %[[T0:.+]] = arith.addi %[[B0]], %[[B1]]
//      CHECK:     %[[T1:.+]] = arith.addi %[[T0]], %[[B1]]
//      CHECK:     %[[T2:.+]] = arith.addi %[[T1]], %[[B2]]
//      CHECK:     %[[T3:.+]] = arith.addi %[[T2]], %[[B3]]
//      CHECK:     %[[T4:.+]] = arith.addi %[[T3]], %[[B3]]
//      CHECK:     %[[T5:.+]] = arith.addi %[[T4]], %[[B4]]
//      CHECK:     linalg.yield %[[T5]]
//      CHECK:  return %[[RETURN]]

// -----

// Drop redundant results.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @drop_redundant_results(
    %arg0 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1, %1 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func @drop_redundant_results
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[ARG0]] :
//      CHECK:   return %[[GENERIC]]
