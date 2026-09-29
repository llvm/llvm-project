// RUN: yaml2obj %S/inputs/or.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "or_i32") (result i32)
    i32.const 10
    i32.const 3
    i32.or)

    (func (export "or_i64") (result i64)
    i64.const 10
    i64.const 3
    i64.or)
)
*/

// CHECK-LABEL: wasmssa.func exported @or_i32() -> i32 {
// CHECK:    %0 = wasmssa.const 10 : i32
// CHECK:    %1 = wasmssa.const 3 : i32
// CHECK:    %2 = wasmssa.or %0 %1 : i32
// CHECK:    wasmssa.return %2 : i32

// CHECK-LABEL: wasmssa.func exported @or_i64() -> i64 {
// CHECK:    %0 = wasmssa.const 10 : i64
// CHECK:    %1 = wasmssa.const 3 : i64
// CHECK:    %2 = wasmssa.or %0 %1 : i64
// CHECK:    wasmssa.return %2 : i64
