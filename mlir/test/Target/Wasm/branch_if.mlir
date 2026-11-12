// RUN: yaml2obj %S/inputs/branch_if.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (type $produce_i32 (func (result i32)))
  (func (type $produce_i32)
    (block $my_block (type $produce_i32)
      i32.const 1
      i32.const 2
      br_if $my_block
      i32.const 1
      i32.add
    )
  )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK:           wasmssa.block : {
// CHECK:             %[[VAL_0:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_1:.*]] = wasmssa.const 2 : i32
// CHECK:             wasmssa.branch_if %[[VAL_1]] to level 0 with args(%[[VAL_0]] : i32) else ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_2:.*]] = wasmssa.const 1 : i32
// CHECK:             %[[VAL_3:.*]] = wasmssa.add %[[VAL_0]] %[[VAL_2]] : i32
// CHECK:             wasmssa.block_return %[[VAL_3]] : i32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_4:.*]]: i32):
// CHECK:           wasmssa.return %[[VAL_4]] : i32
