// RUN: yaml2obj %S/inputs/sub.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func $sub_i32 (result i32)
        i32.const 12
        i32.const 50
        i32.sub
    )

    (func $sub_i64 (result i64)
        i64.const 20
        i64.const 5
        i64.sub
    )

    (func $sub_f32 (result f32)
        f32.const 5
        f32.const 14
        f32.sub
    )

    (func $sub_f64 (result f64)
        f64.const 17
        f64.const 0
        f64.sub
    )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.sub %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_1() -> i64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 20 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 5 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.sub %[[VAL_0]] %[[VAL_1]] : i64
// CHECK:           wasmssa.return %[[VAL_2]] : i64

// CHECK-LABEL:   wasmssa.func @func_2() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.sub %[[VAL_0]] %[[VAL_1]] : f32
// CHECK:           wasmssa.return %[[VAL_2]] : f32

// CHECK-LABEL:   wasmssa.func @func_3() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.700000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.sub %[[VAL_0]] %[[VAL_1]] : f64
// CHECK:           wasmssa.return %[[VAL_2]] : f64
