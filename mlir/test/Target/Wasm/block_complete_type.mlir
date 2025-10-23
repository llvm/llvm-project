// RUN: yaml2obj %S/inputs/block_complete_type.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (type (;0;) (func (param i32) (result i32)))
  (type (;1;) (func (result i32)))
  (func (;0;) (type 1) (result i32)
    i32.const 14
    block (param i32) (result i32)  ;; label = @1
      i32.const 1
      i32.add
    end))
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 14 : i32
// CHECK:           wasmssa.block(%[[VAL_0]]) : i32 : {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32):
// CHECK:             %[[VAL_2:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_3:.*]] = wasmssa.add %[[VAL_1]] %[[VAL_2]] : i32
// CHECK:             wasmssa.block_return %[[VAL_3]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_4:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_4]] : i32
