// RUN: yaml2obj %S/inputs/extend.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (func $i32_s (result i64)
    i32.const 10
    i64.extend_i32_s
  )
  (func $i32_u (result i64)
    i32.const 10
    i64.extend_i32_u
  )
  (func $extend8_32 (result i32)
    i32.const 10
    i32.extend8_s
  )
  (func $extend16_32 (result i32)
    i32.const 10
    i32.extend16_s
  )
  (func $extend8_64 (result i64)
    i64.const 10
    i64.extend8_s
  )
  (func $extend16_64 (result i64)
    i64.const 10
    i64.extend16_s
  )
  (func $extend32_64 (result i64)
    i64.const 10
    i64.extend32_s
  )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend_i32_s %[[VAL_0]] to i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func @func_1() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend_i32_u %[[VAL_0]] to i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func @func_2() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend 8 : ui32 low bits from %[[VAL_0]] : i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func @func_3() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend 16 : ui32 low bits from %[[VAL_0]] : i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func @func_4() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend 8 : ui32 low bits from %[[VAL_0]] : i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func @func_5() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend 16 : ui32 low bits from %[[VAL_0]] : i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func @func_6() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.extend 32 : ui32 low bits from %[[VAL_0]] : i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64
