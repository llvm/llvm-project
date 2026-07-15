// RUN: yaml2obj %S/inputs/reinterpret.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/*
Test generated from:
(module
    (func (export "i32.reinterpret_f32") (result i32)
        f32.const -1
        i32.reinterpret_f32
    )

    (func (export "i64.reinterpret_f64") (result i64)
        f64.const -1
        i64.reinterpret_f64
    )

    (func (export "f32.reinterpret_i32") (result f32)
        i32.const -1
        f32.reinterpret_i32
    )

    (func (export "f64.reinterpret_i64") (result f64)
        i64.const -1
        f64.reinterpret_i64
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @i32.reinterpret_f32() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const -1.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.reinterpret %[[VAL_0]] : f32 as i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32

// CHECK-LABEL:   wasmssa.func exported @i64.reinterpret_f64() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const -1.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.reinterpret %[[VAL_0]] : f64 as i64
// CHECK:           wasmssa.return %[[VAL_1]] : i64

// CHECK-LABEL:   wasmssa.func exported @f32.reinterpret_i32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const -1 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.reinterpret %[[VAL_0]] : i32 as f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @f64.reinterpret_i64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const -1 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.reinterpret %[[VAL_0]] : i64 as f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
