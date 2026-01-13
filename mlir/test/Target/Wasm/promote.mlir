// RUN: yaml2obj %S/inputs/promote.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
  (func $main (result f64)
    f32.const 10.5
    f64.promote_f32
  )
)*/

// CHECK-LABEL:   wasmssa.func @func_0() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.050000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.promote %[[VAL_0]] : f32 to f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
