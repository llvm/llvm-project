// RUN: yaml2obj %S/inputs/global.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s
/* Source code used to create this test:
(module

;; import a global variable from js
(global $imported_glob (import "env" "from_js") i32)

;; create a global variable
(global $normal_glob i32(i32.const 10))
(global $glob_mut (mut i32) (i32.const 10))
(global $glob_mut_ext (mut i32) (i32.const 10))

(global $normal_glob_i64 i64(i64.const 11))
(global $normal_glob_f32 f32(f32.const 12))
(global $normal_glob_f64 f64(f64.const 13))

(func $main (result i32)
;; load both global variables onto the stack
global.get $imported_glob
global.get $normal_glob

i32.add ;; add up both globals

global.get $glob_mut
global.get $glob_mut_ext
i32.add
i32.add
)
)
*/

// CHECK-LABEL:   wasmssa.import_global "from_js" from "env" as @global_0 : i32

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.global_get @global_0 : i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.global_get @global_1 : i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.add %[[VAL_0]] %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = wasmssa.global_get @global_2 : i32
// CHECK:           %[[VAL_4:.*]] = wasmssa.global_get @global_3 : i32
// CHECK:           %[[VAL_5:.*]] = wasmssa.add %[[VAL_3]] %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]] = wasmssa.add %[[VAL_2]] %[[VAL_5]] : i32
// CHECK:           wasmssa.return %[[VAL_6]] : i32

// CHECK-LABEL:   wasmssa.global @global_1 i32 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_2 i32 mutable : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_3 i32 mutable : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_4 i64 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 11 : i64
// CHECK:           wasmssa.return %[[VAL_0]] : i64

// CHECK-LABEL:   wasmssa.global @global_5 f32 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.200000e+01 : f32
// CHECK:           wasmssa.return %[[VAL_0]] : f32

// CHECK-LABEL:   wasmssa.global @global_6 f64 : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.300000e+01 : f64
// CHECK:           wasmssa.return %[[VAL_0]] : f64
