// RUN: yaml2obj %S/inputs/eqz.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func (export "eqz_i32") (result i32)
    i32.const 13
    i32.eqz)

    (func (export "eqz_i64") (result i32)
    i64.const 13
    i64.eqz)
)
*/
// CHECK-LABEL:   wasmssa.func exported @eqz_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 13 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.eqz %[[VAL_0]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func exported @eqz_i64() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 13 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.eqz %[[VAL_0]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32
