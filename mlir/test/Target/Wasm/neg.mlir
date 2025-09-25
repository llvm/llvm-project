// RUN: yaml2obj %S/inputs/neg.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "neg_f32") (result f32)
    f32.const 10
    f32.neg)

    (func (export "neg_f64") (result f64)
    f64.const 10
    f64.neg)
)
*/

// CHECK-LABEL:   wasmssa.func @neg_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.neg %[[VAL_0]] : f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func @neg_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.neg %[[VAL_0]] : f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
