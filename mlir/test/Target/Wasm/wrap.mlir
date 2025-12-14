// RUN: yaml2obj %S/inputs/wrap.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func (export "i64_wrap") (param $in i64) (result i32)
    local.get $in
    i32.wrap_i64
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @i64_wrap(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i64>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.wrap %[[VAL_0]] : i64 to i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32
