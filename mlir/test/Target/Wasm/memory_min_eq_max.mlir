// RUN: yaml2obj %S/inputs/memory_min_eq_max.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module (memory 0 0))
*/

// CHECK-LABEL:   wasmssa.memory @mem_0 nested !wasmssa<limit[0: 0]>
