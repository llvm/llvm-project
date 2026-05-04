// RUN: yaml2obj %S/inputs/rotr.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "rotr_i32") (result i32)
    i32.const 10
    i32.const 3
    i32.rotr)

    (func (export "rotr_i64") (result i64)
    i64.const 10
    i64.const 3
    i64.rotr)
)
*/

// CHECK-LABEL: wasmssa.func exported @rotr_i32() -> i32 {
// CHECK:    %0 = wasmssa.const 10 : i32
// CHECK:    %1 = wasmssa.const 3 : i32
// CHECK:    %2 = wasmssa.rotr %0 by %1 bits : i32
// CHECK:    wasmssa.return %2 : i32

// CHECK-LABEL: wasmssa.func exported @rotr_i64() -> i64 {
// CHECK:    %0 = wasmssa.const 10 : i64
// CHECK:    %1 = wasmssa.const 3 : i64
// CHECK:    %2 = wasmssa.rotr %0 by %1 bits : i64
// CHECK:    wasmssa.return %2 : i64
