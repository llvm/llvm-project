// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir | FileCheck %s

// CHECK-LABEL:   func.func @func_1(
// CHECK-SAME:                      %[[ARG0:.*]]: i32,
// CHECK-SAME:                      %[[ARG1:.*]]: i32) -> i32 {
wasmssa.func @func_1(%arg0: !wasmssa<local ref to i32>, %arg1: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i32>
%v0 = wasmssa.local_get %arg0 : ref to i32
%v1 = wasmssa.local_get %arg1 : ref to i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
%0 = wasmssa.add %v0 %v1 : i32
// CHECK:           return %[[VAL_4]] : i32
wasmssa.return %0 : i32
}

// -----

// CHECK-LABEL:   func.func @func_2(
// CHECK-SAME:                      %[[ARG0:.*]]: i64,
// CHECK-SAME:                      %[[ARG1:.*]]: i64) -> i64 {
wasmssa.func @func_2(%arg0: !wasmssa<local ref to i64>, %arg1: !wasmssa<local ref to i64>) -> i64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i64>
%v0 = wasmssa.local_get %arg0 : ref to i64
%v1 = wasmssa.local_get %arg1 : ref to i64
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i64
%0 = wasmssa.add %v0 %v1 : i64
// CHECK:           return %[[VAL_4]] : i64
wasmssa.return %0 : i64
}

// -----

// CHECK-LABEL:   func.func @func_3(
// CHECK-SAME:                      %[[ARG0:.*]]: f32,
// CHECK-SAME:                      %[[ARG1:.*]]: f32) -> f32 {
wasmssa.func @func_3(%arg0: !wasmssa<local ref to f32>, %arg1: !wasmssa<local ref to f32>) -> f32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f32>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<f32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<f32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<f32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<f32>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<f32>
%v0 = wasmssa.local_get %arg0 : ref to f32
%v1 = wasmssa.local_get %arg1 : ref to f32
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f32
%0 = wasmssa.add %v0 %v1 : f32
// CHECK:           return %[[VAL_4]] : f32
wasmssa.return %0 : f32
}

// -----

// CHECK-LABEL:   func.func @func_4(
// CHECK-SAME:                      %[[ARG0:.*]]: f64,
// CHECK-SAME:                      %[[ARG1:.*]]: f64) -> f64 {
wasmssa.func @func_4(%arg0: !wasmssa<local ref to f64>, %arg1: !wasmssa<local ref to f64>) -> f64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<f64>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<f64>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<f64>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<f64>
%v0 = wasmssa.local_get %arg0 : ref to f64
%v1 = wasmssa.local_get %arg1 : ref to f64
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f64
%0 = wasmssa.add %v0 %v1 : f64
// CHECK:           return %[[VAL_4]] : f64
wasmssa.return %0 : f64
}
