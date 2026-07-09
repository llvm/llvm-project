// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s

// CHECK-LABEL:   func.func @ctz_i32(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = math.cttz %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]] : i32
wasmssa.func exported @ctz_i32(%arg0: !wasmssa<local ref to i32>) -> i32 {
    %v0 = wasmssa.local_get %arg0 : ref to i32
    %op = wasmssa.ctz %v0 : i32
    wasmssa.return %op : i32
}

// CHECK-LABEL:   func.func @ctz_i64(
// CHECK-SAME:      %[[ARG0:.*]]: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_2:.*]] = math.cttz %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
wasmssa.func exported @ctz_i64(%arg0: !wasmssa<local ref to i64>) -> i64 {
    %v0 = wasmssa.local_get %arg0 : ref to i64
    %op = wasmssa.ctz %v0 : i64
    wasmssa.return %op : i64
}
