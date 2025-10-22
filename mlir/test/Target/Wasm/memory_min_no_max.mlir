// RUN: yaml2obj %S/inputs/memory_min_no_max.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module (memory 1))
*/

// CHECK-LABEL:  wasmssa.memory @mem_0 !wasmssa<limit[1:]>
