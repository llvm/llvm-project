// RUN: yaml2obj %S/inputs/comparison_ops.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
    (func $lt_si32 (result i32)
        i32.const 12
        i32.const 50
        i32.lt_s
    )
    (func $le_si32 (result i32)
        i32.const 12
        i32.const 50
        i32.le_s
    )
    (func $lt_ui32 (result i32)
        i32.const 12
        i32.const 50
        i32.lt_u
    )
    (func $le_ui32 (result i32)
        i32.const 12
        i32.const 50
        i32.le_u
    )
    (func $gt_si32 (result i32)
        i32.const 12
        i32.const 50
        i32.gt_s
    )
    (func $gt_ui32 (result i32)
        i32.const 12
        i32.const 50
        i32.gt_u
    )
    (func $ge_si32 (result i32)
        i32.const 12
        i32.const 50
        i32.ge_s
    )
    (func $ge_ui32 (result i32)
        i32.const 12
        i32.const 50
        i32.ge_u
    )
    (func $lt_si64 (result i32)
        i64.const 12
        i64.const 50
        i64.lt_s
    )
    (func $le_si64 (result i32)
        i64.const 12
        i64.const 50
        i64.le_s
    )
    (func $lt_ui64 (result i32)
        i64.const 12
        i64.const 50
        i64.lt_u
    )
    (func $le_ui64 (result i32)
        i64.const 12
        i64.const 50
        i64.le_u
    )
    (func $gt_si64 (result i32)
        i64.const 12
        i64.const 50
        i64.gt_s
    )
    (func $gt_ui64 (result i32)
        i64.const 12
        i64.const 50
        i64.gt_u
    )
    (func $ge_si64 (result i32)
        i64.const 12
        i64.const 50
        i64.ge_s
    )
    (func $ge_ui64 (result i32)
        i64.const 12
        i64.const 50
        i64.ge_u
    )
    (func $lt_f32 (result i32)
        f32.const 5
        f32.const 14
        f32.lt
    )
    (func $le_f32 (result i32)
        f32.const 5
        f32.const 14
        f32.le
    )
    (func $gt_f32 (result i32)
        f32.const 5
        f32.const 14
        f32.gt
    )
    (func $ge_f32 (result i32)
        f32.const 5
        f32.const 14
        f32.ge
    )
    (func $lt_f64 (result i32)
        f64.const 5
        f64.const 14
        f64.lt
    )
    (func $le_f64 (result i32)
        f64.const 5
        f64.const 14
        f64.le
    )
    (func $gt_f64 (result i32)
        f64.const 5
        f64.const 14
        f64.gt
    )
    (func $ge_f64 (result i32)
        f64.const 5
        f64.const 14
        f64.ge
    )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt_si %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_1() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.le_si %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_2() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt_ui %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_3() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.le_ui %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_4() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt_si %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_5() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt_ui %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_6() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge_si %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_7() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge_ui %[[VAL_0]] %[[VAL_1]] : i32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_8() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt_si %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_9() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.le_si %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_10() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt_ui %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_11() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.le_ui %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_12() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt_si %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_13() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt_ui %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_14() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge_si %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_15() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 12 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 50 : i64
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge_ui %[[VAL_0]] %[[VAL_1]] : i64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_16() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt %[[VAL_0]] %[[VAL_1]] : f32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_17() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.le %[[VAL_0]] %[[VAL_1]] : f32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_18() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt %[[VAL_0]] %[[VAL_1]] : f32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_19() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge %[[VAL_0]] %[[VAL_1]] : f32 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_20() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.lt %[[VAL_0]] %[[VAL_1]] : f64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_21() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.le %[[VAL_0]] %[[VAL_1]] : f64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_22() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.gt %[[VAL_0]] %[[VAL_1]] : f64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32

// CHECK-LABEL:   wasmssa.func @func_23() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 5.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = wasmssa.const 1.400000e+01 : f64
// CHECK:           %[[VAL_2:.*]] = wasmssa.ge %[[VAL_0]] %[[VAL_1]] : f64 -> i32
// CHECK:           wasmssa.return %[[VAL_2]] : i32
