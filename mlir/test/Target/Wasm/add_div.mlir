// RUN: yaml2obj %S/inputs/add_div.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
 (module $test.wasm
  (type (;0;) (func (param i32) (result i32)))
  (type (;1;) (func (param i32 i32) (result i32)))
  (import "env" "twoTimes" (func $twoTimes (type 0)))
  (func $add (type 1) (param i32 i32) (result i32)
    local.get 0
    call $twoTimes
    local.get 1
    call $twoTimes
    i32.add
    i32.const 2
    i32.div_s)
  (memory (;0;) 2)
  (global $__stack_pointer (mut i32) (i32.const 66560))
  (export "memory" (memory 0))
  (export "add" (func $add)))
*/

// CHECK-LABEL:   wasmssa.import_func "twoTimes" from "env" as @func_0 {type = (i32) -> i32}

// CHECK-LABEL:   wasmssa.func exported @add(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>,
// CHECK-SAME:      %[[ARG1:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.call @func_0(%[[VAL_0]]) : (i32) -> i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.local_get %[[ARG1]] :  ref to i32
// CHECK:           %[[VAL_3:.*]] = wasmssa.call @func_0(%[[VAL_2]]) : (i32) -> i32
// CHECK:           %[[VAL_4:.*]] = wasmssa.add %[[VAL_1]] %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = wasmssa.const 2 : i32
// CHECK:           %[[VAL_6:.*]] = wasmssa.div_si %[[VAL_4]] %[[VAL_5]] : i32
// CHECK:           wasmssa.return %[[VAL_6]] : i32
// CHECK:         }
// CHECK:         wasmssa.memory exported @memory !wasmssa<limit[2:]>

// CHECK-LABEL:   wasmssa.global @global_0 i32 mutable : {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 66560 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
