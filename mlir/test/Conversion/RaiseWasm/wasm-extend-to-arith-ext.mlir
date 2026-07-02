// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

// CHECK-LABEL:   func.func @func_0() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK:           return %[[VAL_1]] : i64
wasmssa.func @func_0() -> i64 {
  %0 = wasmssa.const 10 : i32
  %1 = wasmssa.extend_i32_s %0 to i64
  wasmssa.return %1 : i64
}

// CHECK-LABEL:   func.func @func_1() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = arith.extui %[[VAL_0]] : i32 to i64
// CHECK:           return %[[VAL_1]] : i64
wasmssa.func @func_1() -> i64 {
  %0 = wasmssa.const 10 : i32
  %1 = wasmssa.extend_i32_u %0 to i64
  wasmssa.return %1 : i64
}

// CHECK-LABEL:   func.func @func_2() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i32 to i8
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// CHECK:           return %[[VAL_2]] : i32
wasmssa.func @func_2() -> i32 {
  %0 = wasmssa.const 10 : i32
  %1 = wasmssa.extend 8 low bits from %0: i32
  wasmssa.return %1 : i32
}

// CHECK-LABEL:   func.func @func_3() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i32 to i16
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i16 to i32
// CHECK:           return %[[VAL_2]] : i32
wasmssa.func @func_3() -> i32 {
  %0 = wasmssa.const 10 : i32
  %1 = wasmssa.extend 16 low bits from %0: i32
  wasmssa.return %1 : i32
}

// CHECK-LABEL:   func.func @func_4() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i64 to i8
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i8 to i64
// CHECK:           return %[[VAL_2]] : i64
wasmssa.func @func_4() -> i64 {
  %0 = wasmssa.const 10 : i64
  %1 = wasmssa.extend 8 low bits from %0: i64
  wasmssa.return %1 : i64
}

// CHECK-LABEL:   func.func @func_5() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i64 to i16
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i16 to i64
// CHECK:           return %[[VAL_2]] : i64
wasmssa.func @func_5() -> i64 {
  %0 = wasmssa.const 10 : i64
  %1 = wasmssa.extend 16 low bits from %0: i64
  wasmssa.return %1 : i64
}

// CHECK-LABEL:   func.func @func_6() -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i64 to i32
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i32 to i64
// CHECK:           return %[[VAL_2]] : i64
wasmssa.func @func_6() -> i64 {
  %0 = wasmssa.const 10 : i64
  %1 = wasmssa.extend 32 low bits from %0: i64
  wasmssa.return %1 : i64
}
