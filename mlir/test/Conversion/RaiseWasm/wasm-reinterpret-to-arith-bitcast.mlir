// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s
module {
// CHECK-LABEL:   func.func @i32.reinterpret_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.bitcast %[[VAL_0]] : f32 to i32
// CHECK:           return %[[VAL_1]] : i32
  wasmssa.func @i32.reinterpret_f32() -> i32 {
    %0 = wasmssa.const -1.000000e+00 : f32
    %1 = wasmssa.reinterpret %0 : f32 as i32
    wasmssa.return %1 : i32
  }

// CHECK-LABEL:   func.func @i64.reinterpret_f64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.bitcast %[[VAL_0]] : f64 to i64
// CHECK:           return %[[VAL_1]] : i64
  wasmssa.func @i64.reinterpret_f64() -> i64 {
    %0 = wasmssa.const -1.000000e+00 : f64
    %1 = wasmssa.reinterpret %0 : f64 as i64
    wasmssa.return %1 : i64
  }

// CHECK-LABEL:   func.func @f32.reinterpret_i32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.bitcast %[[VAL_0]] : i32 to f32
// CHECK:           return %[[VAL_1]] : f32
  wasmssa.func @f32.reinterpret_i32() -> f32 {
    %0 = wasmssa.const -1 : i32
    %1 = wasmssa.reinterpret %0 : i32 as f32
    wasmssa.return %1 : f32
  }

// CHECK-LABEL:   func.func @f64.reinterpret_i64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.bitcast %[[VAL_0]] : i64 to f64
// CHECK:           return %[[VAL_1]] : f64
  wasmssa.func @f64.reinterpret_i64() -> f64 {
    %0 = wasmssa.const -1 : i64
    %1 = wasmssa.reinterpret %0 : i64 as f64
    wasmssa.return %1 : f64
  }
}
