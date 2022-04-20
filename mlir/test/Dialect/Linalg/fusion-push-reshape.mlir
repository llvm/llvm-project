// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=push-expanding-reshape -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func @reshape
// CHECK-SAME: (%[[A:.*]]: tensor<?x16xf32>, %[[B:.*]]: tensor<16xf32>, %[[INIT:.*]]: tensor<?x112x16xf32>)
//      CHECK: %[[RI:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2]] : tensor<?x112x16xf32> into tensor<?x16xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]], #[[$MAP2]]],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x16xf32>, tensor<16xf32>) outs(%[[RI]] : tensor<?x16xf32>)
//      CHECK: %[[RR:.*]] = tensor.expand_shape %[[R]] {{\[}}[0, 1], [2]] : tensor<?x16xf32> into tensor<?x112x16xf32>
//      CHECK: return %[[RR]] : tensor<?x112x16xf32>
func.func @reshape(%A: tensor<?x16xf32>, %B: tensor<16xf32>, %init: tensor<?x112x16xf32>) -> tensor<?x112x16xf32> {
  %0 = tensor.expand_shape %A [[0, 1], [2]]
      : tensor<?x16xf32> into tensor<?x112x16xf32>
  %2 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%0, %B : tensor<?x112x16xf32>, tensor<16xf32>)
  outs(%init : tensor<?x112x16xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  
    %s = arith.subf %arg1, %arg2 : f32
    linalg.yield %s : f32
  } -> tensor<?x112x16xf32>
  return %2 : tensor<?x112x16xf32>
}

// -----

// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func @reshape_multiple
// CHECK-SAME: (%[[A:.*]]: tensor<12544x16xf32>, %[[B:.*]]: tensor<12544x16xf32>, %[[C:.*]]: tensor<16xf32>)
//      CHECK: %[[I:.*]] = linalg.init_tensor [112, 112, 16] : tensor<112x112x16xf32>
//      CHECK: %[[RI:.*]] = tensor.collapse_shape %[[I]] {{\[}}[0, 1], [2]] : tensor<112x112x16xf32> into tensor<12544x16xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]], #[[$MAP3]], #[[$MAP2]]],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[A]], %[[B]], %[[C]] : tensor<12544x16xf32>, tensor<12544x16xf32>, tensor<16xf32>) outs(%[[RI]] : tensor<12544x16xf32>)
//      CHECK: %[[RR:.*]] = tensor.expand_shape %[[R]] {{\[}}[0, 1], [2]] : tensor<12544x16xf32> into tensor<112x112x16xf32>
//      CHECK: return %[[RR]] : tensor<112x112x16xf32>
func.func @reshape_multiple(%A: tensor<12544x16xf32>, %B: tensor<12544x16xf32>,
  %C: tensor<16xf32>) -> tensor<112x112x16xf32> {
  %0 = tensor.expand_shape %A [[0, 1], [2]]
      : tensor<12544x16xf32> into tensor<112x112x16xf32>
  %1 = tensor.expand_shape %B [[0, 1], [2]]
      : tensor<12544x16xf32> into tensor<112x112x16xf32>
  %2 = linalg.init_tensor [112, 112, 16] : tensor<112x112x16xf32>
  %3 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%0, %1, %C : tensor<112x112x16xf32>, tensor<112x112x16xf32>, tensor<16xf32>)
  outs(%2 : tensor<112x112x16xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):  
    %s = arith.subf %arg1, %arg2 : f32
    %m = arith.mulf %s, %arg3 : f32
    linalg.yield %m : f32
  } -> tensor<112x112x16xf32>
  return %3 : tensor<112x112x16xf32>
}

// -----

// Negative test, since the second source is broadcasted from d1 we cannot merge
// d0 and d1 dimensions
// CHECK-LABEL: func @reshape_negative
// CHECK: tensor.expand_shape {{.*}} : tensor<12544x16xf32> into tensor<112x112x16xf32>
// CHECK: linalg.generic
// CHECK: } -> tensor<112x112x16xf32>
func.func @reshape_negative(%A: tensor<12544x16xf32>, %B: tensor<112xf32>) -> tensor<112x112x16xf32> {
  %20 = tensor.expand_shape %A [[0, 1], [2]]
      : tensor<12544x16xf32> into tensor<112x112x16xf32>
  %21 = linalg.init_tensor [112, 112, 16] : tensor<112x112x16xf32>
  %22 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%20, %B : tensor<112x112x16xf32>, tensor<112xf32>)
  outs(%21 : tensor<112x112x16xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  
    %s = arith.subf %arg1, %arg2 : f32
    linalg.yield %s : f32
  } -> tensor<112x112x16xf32>
  return %22 : tensor<112x112x16xf32>
}

// -----

func.func @type_correctness(%arg0 : tensor<6x5xi32>, %arg1 : tensor<5xf32>,
    %arg2 : tensor<5xf32>) -> tensor<2x3x5xf32> {
  %cst_6 = arith.constant 1.000000e+00 : f32
  %cst_7 = arith.constant 7.000000e+00 : f32
  %cst_8 = arith.constant 1.1920929E-7 : f32
  %25 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<6x5xi32> into tensor<2x3x5xi32>
  %26 = linalg.init_tensor [2, 3, 5] : tensor<2x3x5xf32>
  %28 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%25, %arg1, %arg2 : tensor<2x3x5xi32>, tensor<5xf32>, tensor<5xf32>)
      outs(%26 : tensor<2x3x5xf32>) {
      ^bb0(%arg6: i32, %arg7: f32, %arg8: f32, %arg9: f32):  
        %29 = arith.sitofp %arg6 : i32 to f32
        %30 = arith.addf %arg7, %cst_8 : f32
        %31 = arith.divf %cst_7, %30 : f32
        %32 = arith.divf %cst_6, %31 : f32
        %33 = arith.mulf %29, %32 : f32
        %34 = arith.addf %33, %arg8 : f32
        linalg.yield %34 : f32
      } -> tensor<2x3x5xf32>
  return %28 : tensor<2x3x5xf32>
}
// CHECK-LABEL: func @type_correctness
//       CHECK:   %[[OP:.+]] = linalg.generic
//  CHECK-SAME:   ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<6x5xi32>, tensor<5xf32>, tensor<5xf32>)
//  CHECK-SAME:   outs(%{{.+}} : tensor<6x5xf32>)
//       CHECK:   tensor.expand_shape %[[OP]]
//  CHECK-SAME:   tensor<6x5xf32> into tensor<2x3x5xf32>

// -----

func.func @generic_op_index_semantics(%A: tensor<?x16xi64>, %B: tensor<16xi64>, %init: tensor<?x112x16xi64>) -> tensor<?x112x16xi64> {
  %0 = tensor.expand_shape %A [[0, 1], [2]]
      : tensor<?x16xi64> into tensor<?x112x16xi64>
  %2 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
  ins(%0, %B : tensor<?x112x16xi64>, tensor<16xi64>)
  outs(%init : tensor<?x112x16xi64>) {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64):  // no predecessors
    %index = linalg.index 0 : index
    %1 = arith.index_cast %index : index to i64
    %add = arith.addi %arg1, %1 : i64
    %s = arith.subi %add, %arg2 : i64
    linalg.yield %s : i64
  } -> tensor<?x112x16xi64>
  return %2 : tensor<?x112x16xi64>
}
//      CHECK: func @generic_op_index_semantics
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x16xi64>
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[ARG0]]
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]]
//      CHECK:   return %[[RESULT]]
