// RUN: mlir-opt %s --test-vector-break-down-reduction-patterns --cse | FileCheck %s

// NOTE: This test pass is set break down vector reductions of size 2 or fewer.

// CHECK-LABEL:   func.func @reduce_2x_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<2xf32>) -> (f32, f32, f32, f32, f32, f32) {
// CHECK-DAG:      %[[E0:.+]] = vector.extract %[[ARG0]][0] : f32 from vector<2xf32>
// CHECK-DAG:      %[[E1:.+]] = vector.extract %[[ARG0]][1] : f32 from vector<2xf32>
// CHECK-DAG:      %[[R0:.+]] = arith.addf %[[E0]], %[[E1]] : f32
// CHECK-DAG:      %[[R1:.+]] = arith.mulf %[[E0]], %[[E1]] : f32
// CHECK-DAG:      %[[R2:.+]] = arith.minnumf %[[E0]], %[[E1]] : f32
// CHECK-DAG:      %[[R3:.+]] = arith.maxnumf %[[E0]], %[[E1]] : f32
// CHECK-DAG:      %[[R4:.+]] = arith.minimumf %[[E0]], %[[E1]] : f32
// CHECK-DAG:      %[[R5:.+]] = arith.maximumf %[[E0]], %[[E1]] : f32
// CHECK:          return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]]
func.func @reduce_2x_f32(%arg0: vector<2xf32>) -> (f32, f32, f32, f32, f32, f32) {
  %0 = vector.reduction <add>, %arg0 : vector<2xf32> into f32
  %1 = vector.reduction <mul>, %arg0 : vector<2xf32> into f32
  %2 = vector.reduction <minnumf>, %arg0 : vector<2xf32> into f32
  %3 = vector.reduction <maxnumf>, %arg0 : vector<2xf32> into f32
  %4 = vector.reduction <minimumf>, %arg0 : vector<2xf32> into f32
  %5 = vector.reduction <maximumf>, %arg0 : vector<2xf32> into f32
  return %0, %1, %2, %3, %4, %5 : f32, f32, f32, f32, f32, f32
}

// CHECK-LABEL:   func.func @reduce_2x_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<2xi32>) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
// CHECK-DAG:      %[[E0:.+]] = vector.extract %[[ARG0]][0] : i32 from vector<2xi32>
// CHECK-DAG:      %[[E1:.+]] = vector.extract %[[ARG0]][1] : i32 from vector<2xi32>
// CHECK-DAG:      %[[R0:.+]] = arith.addi %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R1:.+]] = arith.muli %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R2:.+]] = arith.minsi %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R3:.+]] = arith.maxsi %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R4:.+]] = arith.minui %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R5:.+]] = arith.maxui %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R6:.+]] = arith.andi %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R7:.+]] = arith.ori %[[E0]], %[[E1]] : i32
// CHECK-DAG:      %[[R8:.+]] = arith.xori %[[E0]], %[[E1]] : i32
// CHECK:          return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]], %[[R8]]
func.func @reduce_2x_i32(%arg0: vector<2xi32>) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
  %0 = vector.reduction <add>, %arg0 : vector<2xi32> into i32
  %1 = vector.reduction <mul>, %arg0 : vector<2xi32> into i32
  %2 = vector.reduction <minsi>, %arg0 : vector<2xi32> into i32
  %3 = vector.reduction <maxsi>, %arg0 : vector<2xi32> into i32
  %4 = vector.reduction <minui>, %arg0 : vector<2xi32> into i32
  %5 = vector.reduction <maxui>, %arg0 : vector<2xi32> into i32
  %6 = vector.reduction <and>, %arg0 : vector<2xi32> into i32
  %7 = vector.reduction <or>, %arg0 : vector<2xi32> into i32
  %8 = vector.reduction <xor>, %arg0 : vector<2xi32> into i32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8 : i32, i32, i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL:   func.func @reduce_1x_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<1xf32>) -> f32 {
// CHECK-NEXT:     %[[E0:.+]] = vector.extract %[[ARG0]][0] : f32 from vector<1xf32>
// CHECK-NEXT:     return %[[E0]] : f32
func.func @reduce_1x_f32(%arg0: vector<1xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<1xf32> into f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @reduce_1x_acc_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<1xf32>, %[[ARG1:.+]]: f32) -> f32 {
// CHECK-NEXT:     %[[E0:.+]] = vector.extract %[[ARG0]][0] : f32 from vector<1xf32>
// CHECK-NEXT:     %[[R0:.+]] = arith.addf %[[E0]], %[[ARG1]] : f32
// CHECK-NEXT:     return %[[R0]] : f32
func.func @reduce_1x_acc_f32(%arg0: vector<1xf32>, %arg1: f32) -> f32 {
  %0 = vector.reduction <add>, %arg0, %arg1 : vector<1xf32> into f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @reduce_1x_acc_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<1xi32>, %[[ARG1:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[E0:.+]] = vector.extract %[[ARG0]][0] : i32 from vector<1xi32>
// CHECK-NEXT:     %[[R0:.+]] = arith.addi %[[E0]], %[[ARG1]] : i32
// CHECK-NEXT:     return %[[R0]] : i32
func.func @reduce_1x_acc_i32(%arg0: vector<1xi32>, %arg1: i32) -> i32 {
  %0 = vector.reduction <add>, %arg0, %arg1 : vector<1xi32> into i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @reduce_2x_acc_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<2xf32>, %[[ARG1:.+]]: f32) -> (f32, f32) {
// CHECK-DAG:      %[[E0:.+]] = vector.extract %[[ARG0]][0] : f32 from vector<2xf32>
// CHECK-DAG:      %[[E1:.+]] = vector.extract %[[ARG0]][1] : f32 from vector<2xf32>
// CHECK:          %[[A0:.+]] = arith.addf %[[E0]], %[[E1]] : f32
// CHECK:          %[[R0:.+]] = arith.addf %[[A0]], %[[ARG1]] : f32
// CHECK:          %[[M0:.+]] = arith.mulf %[[E0]], %[[E1]] fastmath<nnan> : f32
// CHECK:          %[[R1:.+]] = arith.mulf %[[M0]], %[[ARG1]] fastmath<nnan> : f32
// CHECK-NEXT:     return %[[R0]], %[[R1]] : f32, f32
func.func @reduce_2x_acc_f32(%arg0: vector<2xf32>, %arg1: f32) -> (f32, f32) {
  %0 = vector.reduction <add>, %arg0, %arg1 : vector<2xf32> into f32
  %1 = vector.reduction <mul>, %arg0, %arg1 fastmath<nnan> : vector<2xf32> into f32
  return %0, %1 : f32, f32
}

// CHECK-LABEL:   func.func @reduce_3x_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<3xf32>) -> f32 {
// CHECK-NEXT:     %[[R0:.+]] = vector.reduction <add>, %[[ARG0]] : vector<3xf32> into f32
// CHECK-NEXT:     return %[[R0]] : f32
func.func @reduce_3x_f32(%arg0: vector<3xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<3xf32> into f32
  return %0 : f32
}

// Masking is not handled yet.
// CHECK-LABEL:   func.func @reduce_mask_3x_f32
// CHECK-NEXT:     %[[M:.+]] = vector.create_mask
// CHECK-NEXT:     %[[R:.+]] = vector.mask %[[M]]
// CHECK-SAME:       vector.reduction <add>
// CHECK-NEXT:     return %[[R]] : f32
func.func @reduce_mask_3x_f32(%arg0: vector<3xf32>, %arg1: index) -> f32 {
  %mask = vector.create_mask %arg1 : vector<3xi1>
  %0 = vector.mask %mask { vector.reduction <add>, %arg0 : vector<3xf32> into f32 } : vector<3xi1> -> f32
  return %0 : f32
}

// Scalable vectors are not supported.
// CHECK-LABEL:   func.func @reduce_scalable_f32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<[1]xf32>) -> f32 {
// CHECK-NEXT:     %[[R0:.+]] = vector.reduction <add>, %[[ARG0]] : vector<[1]xf32> into f32
// CHECK-NEXT:     return %[[R0]] : f32
func.func @reduce_scalable_f32(%arg0: vector<[1]xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<[1]xf32> into f32
  return %0 : f32
}
