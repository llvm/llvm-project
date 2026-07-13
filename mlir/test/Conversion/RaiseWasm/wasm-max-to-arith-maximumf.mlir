// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s

// CHECK-LABEL:   func.func @max_f32(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f32>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<f32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<f32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<f32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<f32>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<f32>
// CHECK:           %[[VAL_4:.*]] = arith.maximumf %[[VAL_2]], %[[VAL_3]] : f32
// CHECK:           return %[[VAL_4]] : f32
wasmssa.func exported @max_f32(%arg0: !wasmssa<local ref to f32>, %arg1: !wasmssa<local ref to f32>) -> f32 {
    %v0 = wasmssa.local_get %arg0 : ref to f32
    %v1 = wasmssa.local_get %arg1 : ref to f32
    %op = wasmssa.max %v0 %v1 : f32
    wasmssa.return %op : f32
}

// CHECK-LABEL:   func.func @max_f64(
// CHECK-SAME:      %[[ARG0:.*]]: f64,
// CHECK-SAME:      %[[ARG1:.*]]: f64) -> f64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<f64>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<f64>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<f64>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<f64>
// CHECK:           %[[VAL_4:.*]] = arith.maximumf %[[VAL_2]], %[[VAL_3]] : f64
// CHECK:           return %[[VAL_4]] : f64
wasmssa.func exported @max_f64(%arg0: !wasmssa<local ref to f64>, %arg1: !wasmssa<local ref to f64>) -> f64 {
    %v0 = wasmssa.local_get %arg0 : ref to f64
    %v1 = wasmssa.local_get %arg1 : ref to f64
    %op = wasmssa.max %v0 %v1 : f64
    wasmssa.return %op : f64
}
