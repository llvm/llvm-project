// RUN: yaml2obj %S/inputs/local.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module
  (func $local_f32 (result f32)
    (local $var1 f32)
    (local $var2 f32)
    f32.const 8.0
    local.set $var1
    local.get $var1
    f32.const 12.0
    local.tee $var2
    f32.add
  )
  (func $local_i32 (result i32)
    (local $var1 i32)
    (local $var2 i32)
    i32.const 8
    local.set $var1
    local.get $var1
    i32.const 12
    local.tee $var2
    i32.add
  )
  (func $local_arg (param $var i32) (result i32)
    i32.const 3
    local.set $var
    local.get $var
  )
)
*/

// CHECK-LABEL:   wasmssa.func nested @func_0() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local of type f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local of type f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 8.000000e+00 : f32
// CHECK:           wasmssa.local_set %[[VAL_0]] :  ref to f32 to %[[VAL_2]] : f32
// CHECK:           %[[VAL_3:.*]] = wasmssa.local_get %[[VAL_0]] :  ref to f32
// CHECK:           %[[VAL_4:.*]] = wasmssa.const 1.200000e+01 : f32
// CHECK:           %[[VAL_5:.*]] = wasmssa.local_tee %[[VAL_1]] :  ref to f32 to %[[VAL_4]] : f32
// CHECK:           %[[VAL_6:.*]] = wasmssa.add %[[VAL_3]] %[[VAL_5]] : f32
// CHECK:           wasmssa.return %[[VAL_6]] : f32

// CHECK-LABEL:   wasmssa.func nested @func_1() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local of type i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local of type i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 8 : i32
// CHECK:           wasmssa.local_set %[[VAL_0]] :  ref to i32 to %[[VAL_2]] : i32
// CHECK:           %[[VAL_3:.*]] = wasmssa.local_get %[[VAL_0]] :  ref to i32
// CHECK:           %[[VAL_4:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_5:.*]] = wasmssa.local_tee %[[VAL_1]] :  ref to i32 to %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]] = wasmssa.add %[[VAL_3]] %[[VAL_5]] : i32
// CHECK:           wasmssa.return %[[VAL_6]] : i32

// CHECK-LABEL:   wasmssa.func nested @func_2(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 3 : i32
// CHECK:           wasmssa.local_set %[[ARG0]] :  ref to i32 to %[[VAL_0]] : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           wasmssa.return %[[VAL_1]] : i32
