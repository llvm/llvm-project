// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s

module {
  wasmssa.func @func_0() -> f32 {
    %0 = wasmssa.const 2.240000e+00 : f64
    %1 = wasmssa.demote %0 : f64 to f32
    wasmssa.return %1 : f32
  }
}

// CHECK-LABEL:   func.func @func_0() -> f32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.240000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.truncf %[[VAL_0]] : f64 to f32
// CHECK:           return %[[VAL_1]] : f32
