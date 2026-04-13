// RUN: yaml2obj %S/inputs/div.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
    (func (export "div_u_i32") (result i32)
        i32.const 10
        i32.const 2
        i32.div_u
    )

    (func (export "div_u_i32_zero") (result i32)
        i32.const 10
        i32.const 0
        i32.div_u
    )

    (func (export "div_s_i32") (result i32)
        i32.const 10
        i32.const 2
        i32.div_s
    )

    (func (export "div_s_i32_zero") (result i32)
        i32.const 10
        i32.const 0
        i32.div_s
    )

    (func (export "div_u_i64") (result i64)
        i64.const 10
        i64.const 2
        i64.div_u
    )

    ;; explode
    (func (export "div_u_i64_zero") (result i64)
        i64.const 10
        i64.const 0
        i64.div_u
    )

    (func (export "div_s_i64") (result i64)
        i64.const 10
        i64.const 2
        i64.div_s
    )

    ;; explode
    (func (export "div_s_i64_zero") (result i64)
        i64.const 10
        i64.const 0
        i64.div_s
    )

    (func (export "div_f32") (result f32)
        f32.const 10
        f32.const 2
        f32.div
    )

    (func (export "div_f64") (result f64)
        f64.const 10
        f64.const 2
        f64.div
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @div_u_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_ui %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func exported @div_u_i32_zero() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_ui %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func exported @div_s_i32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_si %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func exported @div_s_i32_zero() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_si %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func exported @div_u_i64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_ui %[[VAL_0]] %[[VAL_1]] : i64
// CHECK:           wasmssa.return %[[VAL_2]] : i64

// CHECK-LABEL:   wasmssa.func exported @div_u_i64_zero() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_ui %[[VAL_0]] %[[VAL_1]] : i64
// CHECK:           wasmssa.return %[[VAL_2]] : i64

// CHECK-LABEL:   wasmssa.func exported @div_s_i64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_si %[[VAL_0]] %[[VAL_1]] : i64
// CHECK:           wasmssa.return %[[VAL_2]] : i64

// CHECK-LABEL:   wasmssa.func exported @div_s_i64_zero() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.div_si %[[VAL_0]] %[[VAL_1]] : i64
// CHECK:           wasmssa.return %[[VAL_2]] : i64

// CHECK-LABEL:   wasmssa.func exported @div_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.div %[[VAL_0]] %[[VAL_1]] : f32
// CHECK:           wasmssa.return %[[VAL_2]] : f32

// CHECK-LABEL:   wasmssa.func exported @div_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.000000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 2.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.div %[[VAL_0]] %[[VAL_1]] : f64
// CHECK:           wasmssa.return %[[VAL_2]] : f64
