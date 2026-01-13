// RUN: yaml2obj %S/inputs/loop.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* IR generated from:
(module
  (func
    (loop $my_loop
    )
  )
)*/

// CHECK-LABEL:   wasmssa.func @func_0() {
// CHECK:           wasmssa.loop : {
// CHECK:             wasmssa.block_return
// CHECK:           }> ^bb1
// CHECK:         ^bb1:
// CHECK:           wasmssa.return
// CHECK:         }
