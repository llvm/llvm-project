// RUN: yaml2obj %S/inputs/block.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
(func(export "i_am_a_block")
(block $i_am_a_block)
)
)
*/

// CHECK-LABEL:   wasmssa.func exported @i_am_a_block() {
// CHECK:           wasmssa.block : {
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb1
// CHECK:         ^bb1:
// CHECK:           wasmssa.return
