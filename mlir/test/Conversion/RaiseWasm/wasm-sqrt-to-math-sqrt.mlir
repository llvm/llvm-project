// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s

// CHECK-LABEL:   func.func @sqrt_f32(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<f32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<f32>
// CHECK:           %[[VAL_2:.*]] = math.sqrt %[[VAL_1]] : f32
// CHECK:           return %[[VAL_2]] : f32
wasmssa.func @sqrt_f32(%arg0: !wasmssa<local ref to f32>) -> f32 {
    %local = wasmssa.local_get %arg0 : ref to f32
    %op = wasmssa.sqrt %local : f32
    wasmssa.return %op : f32
}

// CHECK-LABEL:   func.func @sqrt_f64(
// CHECK-SAME:      %[[ARG0:.*]]: f64) -> f64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<f64>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<f64>
// CHECK:           %[[VAL_2:.*]] = math.sqrt %[[VAL_1]] : f64
// CHECK:           return %[[VAL_2]] : f64
wasmssa.func @sqrt_f64(%arg0: !wasmssa<local ref to f64>) -> f64 {
    %local = wasmssa.local_get %arg0 : ref to f64
    %op = wasmssa.sqrt %local : f64
    wasmssa.return %op : f64
}
