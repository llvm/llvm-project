// RUN: yaml2obj %S/inputs/empty_blocks_list_and_stack.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (func (param $num i32)
    (block $b1
        (block $b2
            (block $b3
            )
        )
    )
  )

  (func (param $num i32)
    (block $b1)
    (block $b2)
    (block $b3)
  )
)

*/

// CHECK-LABEL:   wasmssa.func @func_0(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) {
// CHECK:           wasmssa.block : {
// CHECK:             wasmssa.block : {
// CHECK:               wasmssa.block : {
// CHECK:                 wasmssa.block_return
// CHECK:               }> ^bb1
// CHECK:             ^bb1:
// CHECK:               wasmssa.block_return
// CHECK:             }> ^bb1
// CHECK:           ^bb1:
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb1
// CHECK:         ^bb1:
// CHECK:           wasmssa.return

// CHECK-LABEL:   wasmssa.func @func_1(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) {
// CHECK:           wasmssa.block : {
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb1
// CHECK:         ^bb1:
// CHECK:           wasmssa.block : {
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb2
// CHECK:         ^bb2:
// CHECK:           wasmssa.block : {
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb3
// CHECK:         ^bb3:
// CHECK:           wasmssa.return
