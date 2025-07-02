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
)
*/

// CHECK-LABEL:   wasmssa.import_global "from_js" from "env" as @global_0 nested : i32


// CHECK-LABEL:   wasmssa.global @global_1 i32 nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_2 i32 mutable nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_3 i32 mutable nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 10 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32

// CHECK-LABEL:   wasmssa.global @global_4 i64 nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 11 : i64
// CHECK:           wasmssa.return %[[VAL_0]] : i64

// CHECK-LABEL:   wasmssa.global @global_5 f32 nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.200000e+01 : f32
// CHECK:           wasmssa.return %[[VAL_0]] : f32

// CHECK-LABEL:   wasmssa.global @global_6 f64 nested : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 1.300000e+01 : f64
// CHECK:           wasmssa.return %[[VAL_0]] : f64
