// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

// CHECK-LABEL:   func.func @get_some_const() -> (i32, i64, f32, f64) {
wasmssa.func exported @get_some_const() -> (i32, i64, f32, f64) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 17 : i32
%0 = wasmssa.const 17: i32
// CHECK:           %[[VAL_1:.*]] = arith.constant -163 : i64
%1 = wasmssa.const -163 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 3.140000e+00 : f32
%2 = wasmssa.const 3.14 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant -1.575000e+02 : f64
%3 = wasmssa.const -157.5 : f64
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, i64, f32, f64
wasmssa.return %0, %1, %2, %3 : i32, i64, f32, f64
}
