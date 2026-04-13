// RUN: yaml2obj %S/inputs/block_value_type.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (type (;0;) (func (result i32)))
  (func (;0;) (type 0) (result i32)
    block (result i32)  ;; label = @1
      i32.const 17
    end))
*/


// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           wasmssa.block : {
// CHECK:             %[[VAL_0:.*]] = wasmssa.const 17 : i32
// CHECK:             wasmssa.block_return %[[VAL_0]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_1:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_1]] : i32
