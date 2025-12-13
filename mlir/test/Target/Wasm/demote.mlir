// RUN: yaml2obj %S/inputs/demote.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (func $main (result f32)
    f64.const 2.24
    f32.demote_f64
    )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 2.240000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.demote %[[VAL_0]] : f64 to f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32
