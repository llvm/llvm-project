// RUN: yaml2obj %S/inputs/eq.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func $eq_i32 (result i32)
        i32.const 12
        i32.const 50
        i32.eq
    )

    (func $eq_i64 (result i32)
        i64.const 20
        i64.const 5
        i64.eq
    )

    (func $eq_f32 (result i32)
        f32.const 5
        f32.const 14
        f32.eq
    )

    (func $eq_f64 (result i32)
        f64.const 17
        f64.const 0
        f64.eq
    )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.eq %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func @func_1() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 20 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 5 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.eq %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func @func_2() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.eq %[[VAL_0]] %[[VAL_1]] : f32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32
// CHECK:         }

// CHECK-LABEL:   wasmssa.func @func_3() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.700000e+01 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.eq %[[VAL_0]] %[[VAL_1]] : f64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32
// CHECK:         }
