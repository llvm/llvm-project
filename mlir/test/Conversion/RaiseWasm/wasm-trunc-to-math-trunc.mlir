// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s

module {
  wasmssa.func @func_4() -> f64 {
    %0 = wasmssa.const -1.210000e+01 : f64
    %1 = wasmssa.trunc %0 : f64
    wasmssa.return %1 : f64
  }
  wasmssa.func @func_5() -> f32 {
    %0 = wasmssa.const 1.618000e+00 : f32
    %1 = wasmssa.trunc %0 : f32
    wasmssa.return %1 : f32
  }
}

// CHECK-LABEL:   func.func @func_4() -> f64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1.210000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = math.trunc %[[VAL_0]] : f64
// CHECK:           return %[[VAL_1]] : f64

// CHECK-LABEL:   func.func @func_5() -> f32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.618000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = math.trunc %[[VAL_0]] : f32
// CHECK:           return %[[VAL_1]] : f32
