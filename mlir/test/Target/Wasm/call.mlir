// RUN: yaml2obj %S/inputs/call.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
(func $forty_two (result i32)
i32.const 42)
(func(export "forty_two")(result i32)
call $forty_two))
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 42 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.func exported @forty_two() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.call @func_0 : () -> i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
