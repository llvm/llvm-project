// RUN: yaml2obj %S/inputs/abs.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "abs_f32") (result f32)
    f32.const 10
    f32.abs)

    (func (export "abs_f64") (result f64)
    f64.const 10
    f64.abs)
)
*/

// CHECK-LABEL:   wasmssa.func exported @abs_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.abs %[[VAL_0]] : f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @abs_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.abs %[[VAL_0]] : f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
