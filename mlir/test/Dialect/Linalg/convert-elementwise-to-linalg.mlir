// RUN: mlir-opt -convert-elementwise-to-linalg -split-input-file %s | FileCheck %s

// In-depth checking of the linalg.generic op for a very trivial case.
// CHECK: #map = affine_map<() -> ()>
// CHECK-LABEL:   func @addf_rank0
func @addf_rank0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %{{.*}} = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>) {
  // CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
  // CHECK:   %[[YIELD:.*]] = addf %[[LHS]], %[[RHS]] : f32
  // CHECK:   linalg.yield %[[YIELD]] : f32
  // CHECK: } -> tensor<f32>
  %0 = addf %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Check indexing maps and iterator types for the rank > 0 case.
// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @addf_rank1
func @addf_rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic{{.*}}indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
  %0 = addf %arg0, %arg1 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Check a unary op.
// CHECK-LABEL: func @exp
func @exp(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[SCALAR:.*]]: f32):
  // CHECK:   %[[YIELD:.*]] = exp %[[SCALAR]] : f32
  // CHECK:   linalg.yield %[[YIELD]] : f32
  %0 = exp %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Check a case with varying operand types.
// CHECK-LABEL: func @select
func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[PRED:.*]]: i1, %[[TRUE_VAL:.*]]: i32, %[[FALSE_VAL:.*]]: i32):
  // CHECK:   select %[[PRED]], %[[TRUE_VAL]], %[[FALSE_VAL]] : i32
  %0 = select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Spot-check an op that requires copying attributes properly to the created scalar op.
// CHECK-LABEL: func @cmpf(
func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  // CHECK: cmpf "olt", %{{.*}}, %{{.*}} : f32
  %0 = cmpf "olt", %arg0, %arg1 : tensor<f32>
  return %0 : tensor<i1>
}
