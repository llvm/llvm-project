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

func.func @arith_select(%arg0: i1, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> () {
  // CHECK: [[V0:[^ ]*]] = emitc.conditional %arg0, %arg1, %arg2 : tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : i1, tensor<8xi32>
  return
}
