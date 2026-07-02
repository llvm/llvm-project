// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

module {
  wasmssa.func @eqz_i32() -> i32 {
    %0 = wasmssa.const 13 : i32
    %1 = wasmssa.eqz %0 : i32 -> i32
    wasmssa.return %1 : i32
  }
  wasmssa.func @eqz_i64() -> i32 {
    %0 = wasmssa.const 13 : i64
    %1 = wasmssa.eqz %0 : i64 -> i32
    wasmssa.return %1 : i32
  }
}

// CHECK-LABEL:   func.func @eqz_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 13 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32

// CHECK-LABEL:   func.func @eqz_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 13 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i32
// CHECK:           return %[[VAL_3]] : i32
