// RUN: yaml2obj %S/inputs/shr_u.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "shr_u_i32") (result i32)
    i32.const 10
    i32.const 3
    i32.shr_u)

    (func (export "shr_u_i64") (result i64)
    i64.const 10
    i64.const 3
    i64.shr_u)
)
*/

// CHECK-LABEL: wasmssa.func exported @shr_u_i32() -> i32 {
// CHECK:    %0 = wasmssa.const 10 : i32
// CHECK:    %1 = wasmssa.const 3 : i32
// CHECK:    %2 = wasmssa.shr_u %0 by %1 bits : i32
// CHECK:    wasmssa.return %2 : i32

// CHECK-LABEL: wasmssa.func exported @shr_u_i64() -> i64 {
// CHECK:    %0 = wasmssa.const 10 : i64
// CHECK:    %1 = wasmssa.const 3 : i64
// CHECK:    %2 = wasmssa.shr_u %0 by %1 bits : i64
// CHECK:    wasmssa.return %2 : i64
