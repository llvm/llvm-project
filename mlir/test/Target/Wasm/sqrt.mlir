// RUN: yaml2obj %S/inputs/sqrt.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "sqrt_f32") (result f32)
    f32.const 10
    f32.sqrt)

    (func (export "sqrt_f64") (result f64)
    f64.const 10
    f64.sqrt)
)
*/

// CHECK-LABEL:   wasmssa.func exported @sqrt_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.sqrt %[[VAL_0]] : f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @sqrt_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.sqrt %[[VAL_0]] : f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
