// RUN: mlir-opt -split-input-file -convert-arith-to-emitc %s | FileCheck %s

// CHECK-LABEL: arith_constants
func.func @arith_constants() {
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : index
  %c_index = arith.constant 0 : index
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : i32
  %c_signless_int_32 = arith.constant 0 : i32
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0.{{0+}}e+00 : f32
  %c_float_32 = arith.constant 0.0 : f32
  // CHECK: emitc.constant
  // CHECK-SAME: value = dense<0> : tensor<i32>
  %c_tensor_single_value = arith.constant dense<0> : tensor<i32>
  // CHECK: emitc.constant
  // CHECK-SAME: value{{.*}}[1, 2], [-3, 9], [0, 0], [2, -1]{{.*}}tensor<4x2xi64>
  %c_tensor_value = arith.constant dense<[[1, 2], [-3, 9], [0, 0], [2, -1]]> : tensor<4x2xi64>
  return
}

// -----

func.func @arith_ops(%arg0: f32, %arg1: f32) {
  // CHECK: [[V0:[^ ]*]] = emitc.add %arg0, %arg1 : (f32, f32) -> f32
  %0 = arith.addf %arg0, %arg1 : f32
  // CHECK: [[V1:[^ ]*]] = emitc.div %arg0, %arg1 : (f32, f32) -> f32
  %1 = arith.divf %arg0, %arg1 : f32  
  // CHECK: [[V2:[^ ]*]] = emitc.mul %arg0, %arg1 : (f32, f32) -> f32
  %2 = arith.mulf %arg0, %arg1 : f32
  // CHECK: [[V3:[^ ]*]] = emitc.sub %arg0, %arg1 : (f32, f32) -> f32
  %3 = arith.subf %arg0, %arg1 : f32

  return
}

// -----

// CHECK-LABEL: arith_integer_ops
func.func @arith_integer_ops(%arg0: i32, %arg1: i32) {
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[ADD:[^ ]*]] = emitc.add %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[ADD]] : ui32 to i32
  %0 = arith.addi %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[SUB:[^ ]*]] = emitc.sub %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[SUB]] : ui32 to i32
  %1 = arith.subi %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[MUL:[^ ]*]] = emitc.mul %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[MUL]] : ui32 to i32
  %2 = arith.muli %arg0, %arg1 : i32

  return
}

// -----

// CHECK-LABEL: arith_integer_ops_signed_nsw
func.func @arith_integer_ops_signed_nsw(%arg0: i32, %arg1: i32) {
  // CHECK: emitc.add %arg0, %arg1 : (i32, i32) -> i32
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : i32
  // CHECK: emitc.sub %arg0, %arg1 : (i32, i32) -> i32
  %1 = arith.subi %arg0, %arg1 overflow<nsw>  : i32
  // CHECK: emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %2 = arith.muli %arg0, %arg1 overflow<nsw> : i32

  return
}

// -----

// CHECK-LABEL: arith_index
func.func @arith_index(%arg0: index, %arg1: index) {
  // CHECK: emitc.add %arg0, %arg1 : (index, index) -> index
  %0 = arith.addi %arg0, %arg1 : index
  // CHECK: emitc.sub %arg0, %arg1 : (index, index) -> index
  %1 = arith.subi %arg0, %arg1 : index
  // CHECK: emitc.mul %arg0, %arg1 : (index, index) -> index
  %2 = arith.muli %arg0, %arg1 : index

  return
}

// -----

func.func @arith_select(%arg0: i1, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> () {
  // CHECK: [[V0:[^ ]*]] = emitc.conditional %arg0, %arg1, %arg2 : tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : i1, tensor<8xi32>
  return
}
