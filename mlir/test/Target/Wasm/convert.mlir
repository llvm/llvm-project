// RUN: yaml2obj %S/inputs/convert.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to generate this test:
(module
    (func (export "convert_i32_u_to_f32") (result f32)
    i32.const 10
    f32.convert_i32_u
    )

    (func (export "convert_i32_s_to_f32") (result f32)
    i32.const 42
    f32.convert_i32_s
    )

    (func (export "convert_i64_u_to_f32") (result f32)
    i64.const 17
    f32.convert_i64_u
    )

    (func (export "convert_i64s_to_f32") (result f32)
    i64.const 10
    f32.convert_i64_s
    )

    (func (export "convert_i32_u_to_f64") (result f64)
    i32.const 10
    f64.convert_i32_u
    )

    (func (export "convert_i32_s_to_f64") (result f64)
    i32.const 42
    f64.convert_i32_s
    )

    (func (export "convert_i64_u_to_f64") (result f64)
    i64.const 17
    f64.convert_i64_u
    )

    (func (export "convert_i64s_to_f64") (result f64)
    i64.const 10
    f64.convert_i64_s
    )
)
*/

// CHECK-LABEL:   wasmssa.func exported @convert_i32_u_to_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_u %[[VAL_0]] : i32 to f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @convert_i32_s_to_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 42 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_s %[[VAL_0]] : i32 to f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @convert_i64_u_to_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 17 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_u %[[VAL_0]] : i64 to f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @convert_i64s_to_f32() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_s %[[VAL_0]] : i64 to f32
// CHECK:           wasmssa.return %[[VAL_1]] : f32

// CHECK-LABEL:   wasmssa.func exported @convert_i32_u_to_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_u %[[VAL_0]] : i32 to f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64

// CHECK-LABEL:   wasmssa.func exported @convert_i32_s_to_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 42 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_s %[[VAL_0]] : i32 to f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64

// CHECK-LABEL:   wasmssa.func exported @convert_i64_u_to_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 17 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_u %[[VAL_0]] : i64 to f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64

// CHECK-LABEL:   wasmssa.func exported @convert_i64s_to_f64() -> f64 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i64
// CHECK:           %[[VAL_1:.*]] = wasmssa.convert_s %[[VAL_0]] : i64 to f64
// CHECK:           wasmssa.return %[[VAL_1]] : f64
