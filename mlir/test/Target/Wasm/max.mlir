// RUN: yaml2obj %S/inputs/max.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "max_f32") (result f32)
    f32.const 10
    f32.const 1
    f32.max
    )

    (func (export "max_f64") (result f64)
    f64.const 10
    f64.const 1
    f64.max
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @min_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.max %[[VAL_0]] %[[VAL_1]] : f32
// CHECK:           wasmssa.return %[[VAL_2]] : f32


// CHECK-LABEL:   wasmssa.func exported @min_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.max %[[VAL_0]] %[[VAL_1]] : f64
// CHECK:           wasmssa.return %[[VAL_2]] : f64
