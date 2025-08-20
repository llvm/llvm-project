// RUN: yaml2obj %S/inputs/popcnt.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "popcnt_i32") (result i32)
    i32.const 10
    i32.popcnt
    )

    (func (export "popcnt_i64") (result i64)
    i64.const 10
    i64.popcnt
    )
)
*/

// CHECK-LABEL:   wasmssa.func @popcnt_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.popcnt %[[VAL_0]] : i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func @popcnt_i64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.popcnt %[[VAL_0]] : i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64
