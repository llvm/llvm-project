// RUN: yaml2obj %S/inputs/trunc_si.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "trunc_si_f32_to_i32") (result i32)
        f32.const 10
        i32.trunc_f32_s
    )

    (func (export "trunc_si_f64_to_i32") (result i32)
        f64.const 10
        i32.trunc_f64_s
    )

    (func (export "trunc_si_f32_to_i64") (result i64)
        f32.const 10
        i64.trunc_f32_s
    )

    (func (export "trunc_si_f64_to_i64") (result i64)
        f64.const 10
        i64.trunc_f64_s
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @trunc_si_f32_to_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.trunc_si %[[VAL_0]] : f32 to i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func exported @trunc_si_f64_to_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.trunc_si %[[VAL_0]] : f64 to i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func exported @trunc_si_f32_to_i64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.trunc_si %[[VAL_0]] : f32 to i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func exported @trunc_si_f64_to_i64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.trunc_si %[[VAL_0]] : f64 to i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64
