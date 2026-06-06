// RUN: mlir-opt %s --test-vector-chained-reduction-folding-patterns | FileCheck %s

// CHECK-LABEL:   func.func @reduce_1x_fp32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xf32>) -> f32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0.0
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ARG0]], %[[CST]] : vector<8xf32> into f32
// CHECK-NEXT:       return %[[RES]] : f32
func.func @reduce_1x_fp32(%arg0: vector<8xf32>) -> f32 {
  %cst0 = arith.constant 0.0 : f32
  %0 = vector.reduction <add>, %arg0, %cst0 : vector<8xf32> into f32
  return %0 : f32
}

// CHECK-LABEL:   func.func @reduce_2x_fp32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xf32>, %[[ARG1:.+]]: vector<8xf32>) -> f32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0.0
// CHECK-DAG:        %[[ADD:.+]] = arith.addf %[[ARG0]], %[[ARG1]] : vector<8xf32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]], %[[CST]] : vector<8xf32> into f32
// CHECK-NEXT:       return %[[RES]] : f32
func.func @reduce_2x_fp32(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> f32 {
  %cst0 = arith.constant 0.0 : f32
  %0 = vector.reduction <add>, %arg0, %cst0 : vector<8xf32> into f32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xf32> into f32
  return %1 : f32
}

// CHECK-LABEL:   func.func @reduce_2x_no_acc_fp32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xf32>, %[[ARG1:.+]]: vector<8xf32>) -> f32 {
// CHECK:            %[[ADD:.+]] = arith.addf %[[ARG0]], %[[ARG1]] : vector<8xf32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]] : vector<8xf32> into f32
// CHECK-NEXT:       return %[[RES]] : f32
func.func @reduce_2x_no_acc_fp32(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> f32 {
  %0 = vector.reduction <add>, %arg0 : vector<8xf32> into f32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xf32> into f32
  return %1 : f32
}

// CHECK-LABEL:   func.func @reduce_2x_zero_add_fp32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xf32>, %[[ARG1:.+]]: vector<8xf32>) -> f32 {
// CHECK:            %[[ADD:.+]] = arith.addf %[[ARG0]], %[[ARG1]] : vector<8xf32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]] : vector<8xf32> into f32
// CHECK-NEXT:       return %[[RES]] : f32
func.func @reduce_2x_zero_add_fp32(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> f32 {
  %cst0 = arith.constant dense<0.0> : vector<8xf32>
  %x = arith.addf %arg0, %cst0 : vector<8xf32>
  %0 = vector.reduction <add>, %x : vector<8xf32> into f32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xf32> into f32
  return %1 : f32
}

// CHECK-LABEL:   func.func @reduce_3x_fp32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xf32>, %[[ARG1:.+]]: vector<8xf32>,
// CHECK-SAME:     %[[ARG2:.+]]: vector<8xf32>) -> f32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0.0
// CHECK-DAG:        %[[ADD0:.+]] = arith.addf %[[ARG1]], %[[ARG2]] : vector<8xf32>
// CHECK-DAG:        %[[ADD1:.+]] = arith.addf %[[ARG0]], %[[ADD0]] : vector<8xf32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD1]], %[[CST]] : vector<8xf32> into f32
// CHECK-NEXT:       return %[[RES]] : f32
func.func @reduce_3x_fp32(%arg0: vector<8xf32>, %arg1: vector<8xf32>,
                          %arg2: vector<8xf32>) -> f32 {
  %cst0 = arith.constant 0.0 : f32
  %0 = vector.reduction <add>, %arg0, %cst0 : vector<8xf32> into f32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xf32> into f32
  %2 = vector.reduction <add>, %arg2, %1 : vector<8xf32> into f32
  return %2 : f32
}

// CHECK-LABEL:   func.func @reduce_1x_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xi32>) -> i32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ARG0]], %[[CST]] : vector<8xi32> into i32
// CHECK-NEXT:       return %[[RES]] : i32
func.func @reduce_1x_i32(%arg0: vector<8xi32>) -> i32 {
  %cst0 = arith.constant 0 : i32
  %0 = vector.reduction <add>, %arg0, %cst0 : vector<8xi32> into i32
  return %0 : i32
}

// CHECK-LABEL:   func.func @reduce_2x_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xi32>, %[[ARG1:.+]]: vector<8xi32>) -> i32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0
// CHECK-DAG:        %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG1]] : vector<8xi32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]], %[[CST]] : vector<8xi32> into i32
// CHECK-NEXT:       return %[[RES]] : i32
func.func @reduce_2x_i32(%arg0: vector<8xi32>, %arg1: vector<8xi32>) -> i32 {
  %cst0 = arith.constant 0 : i32
  %0 = vector.reduction <add>, %arg0, %cst0 : vector<8xi32> into i32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xi32> into i32
  return %1 : i32
}

// CHECK-LABEL:   func.func @reduce_2x_no_acc_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xi32>, %[[ARG1:.+]]: vector<8xi32>) -> i32 {
// CHECK:            %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG1]] : vector<8xi32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]] : vector<8xi32> into i32
// CHECK-NEXT:       return %[[RES]] : i32
func.func @reduce_2x_no_acc_i32(%arg0: vector<8xi32>, %arg1: vector<8xi32>) -> i32 {
  %0 = vector.reduction <add>, %arg0 : vector<8xi32> into i32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xi32> into i32
  return %1 : i32
}

// CHECK-LABEL:   func.func @reduce_2x_zero_add_i32(
// CHECK-SAME:     %[[ARG0:.+]]: vector<8xi32>, %[[ARG1:.+]]: vector<8xi32>) -> i32 {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0
// CHECK-DAG:        %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG1]] : vector<8xi32>
// CHECK-NEXT:       %[[RES:.+]] = vector.reduction <add>, %[[ADD]], %[[CST]] : vector<8xi32> into i32
// CHECK-NEXT:       return %[[RES]] : i32
func.func @reduce_2x_zero_add_i32(%arg0: vector<8xi32>, %arg1: vector<8xi32>) -> i32 {
  %cst0 = arith.constant 0 : i32
  %cstV = arith.constant dense<0> : vector<8xi32>
  %x = arith.addi %arg0, %cstV : vector<8xi32>
  %0 = vector.reduction <add>, %x, %cst0 : vector<8xi32> into i32
  %1 = vector.reduction <add>, %arg1, %0 : vector<8xi32> into i32
  return %1 : i32
}
