// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

module {
  wasmssa.func @i64_wrap(%arg0: !wasmssa<local ref to i64>) -> i32 {
    %0 = wasmssa.local_get %arg0 :  ref to i64
    %1 = wasmssa.wrap %0 : i64 to i32
    wasmssa.return %1 : i32
  }
}

// CHECK-LABEL:   func.func @i64_wrap(
// CHECK-SAME:      %[[ARG0:.*]]: i64) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_2:.*]] = arith.trunci %[[VAL_1]] : i64 to i32
// CHECK:           return %[[VAL_2]] : i32
