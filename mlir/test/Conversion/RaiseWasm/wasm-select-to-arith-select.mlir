// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

// CHECK-LABEL:   func.func @select_i32() -> i32 {
wasmssa.func @select_i32() -> i32 {
// CHECK:           %[[COND:.*]] = arith.constant 1 : i32
  %cond = wasmssa.const 1 : i32
// CHECK:           %[[VAL_A:.*]] = arith.constant 12 : i32
  %a = wasmssa.const 12 : i32
// CHECK:           %[[VAL_B:.*]] = arith.constant 50 : i32
  %b = wasmssa.const 50 : i32
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[FLAG:.*]] = arith.cmpi ne, %[[COND]], %[[ZERO]] : i32
// CHECK:           %[[RES:.*]] = arith.select %[[FLAG]], %[[VAL_A]], %[[VAL_B]] : i32
  %r = wasmssa.select %cond, %a, %b : i32
// CHECK:           return %[[RES]] : i32
  wasmssa.return %r : i32
}

// CHECK-LABEL:   func.func @select_i64() -> i64 {
wasmssa.func @select_i64() -> i64 {
// CHECK:           %[[COND:.*]] = arith.constant 0 : i32
  %cond = wasmssa.const 0 : i32
// CHECK:           %[[VAL_A:.*]] = arith.constant 12 : i64
  %a = wasmssa.const 12 : i64
// CHECK:           %[[VAL_B:.*]] = arith.constant 50 : i64
  %b = wasmssa.const 50 : i64
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[FLAG:.*]] = arith.cmpi ne, %[[COND]], %[[ZERO]] : i32
// CHECK:           %[[RES:.*]] = arith.select %[[FLAG]], %[[VAL_A]], %[[VAL_B]] : i64
  %r = wasmssa.select %cond, %a, %b : i64
// CHECK:           return %[[RES]] : i64
  wasmssa.return %r : i64
}

// CHECK-LABEL:   func.func @select_f32() -> f32 {
wasmssa.func @select_f32() -> f32 {
// CHECK:           %[[COND:.*]] = arith.constant 1 : i32
  %cond = wasmssa.const 1 : i32
// CHECK:           %[[VAL_A:.*]] = arith.constant 1.250000e-01 : f32
  %a = wasmssa.const 0.125 : f32
// CHECK:           %[[VAL_B:.*]] = arith.constant 2.500000e-01 : f32
  %b = wasmssa.const 0.25 : f32
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[FLAG:.*]] = arith.cmpi ne, %[[COND]], %[[ZERO]] : i32
// CHECK:           %[[RES:.*]] = arith.select %[[FLAG]], %[[VAL_A]], %[[VAL_B]] : f32
  %r = wasmssa.select %cond, %a, %b : f32
// CHECK:           return %[[RES]] : f32
  wasmssa.return %r : f32
}

// CHECK-LABEL:   func.func @select_f64() -> f64 {
wasmssa.func @select_f64() -> f64 {
// CHECK:           %[[COND:.*]] = arith.constant 0 : i32
  %cond = wasmssa.const 0 : i32
// CHECK:           %[[VAL_A:.*]] = arith.constant 3.140000e+00 : f64
  %a = wasmssa.const 3.14 : f64
// CHECK:           %[[VAL_B:.*]] = arith.constant 2.718000e+00 : f64
  %b = wasmssa.const 2.718 : f64
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[FLAG:.*]] = arith.cmpi ne, %[[COND]], %[[ZERO]] : i32
// CHECK:           %[[RES:.*]] = arith.select %[[FLAG]], %[[VAL_A]], %[[VAL_B]] : f64
  %r = wasmssa.select %cond, %a, %b : f64
// CHECK:           return %[[RES]] : f64
  wasmssa.return %r : f64
}
