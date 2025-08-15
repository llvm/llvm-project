// RUN: yaml2obj %S/inputs/stats.yaml.wasm -o - | mlir-translate --import-wasm -stats 2>&1 | FileCheck %s
// Check that we get the correct stats for a module that has a single
// function, table, memory, and global.
// REQUIRES: asserts

/* Source code used to create this test:
(module
  (type (;0;) (func (param i32) (result i32)))
  (func (;0;) (type 0) (param i32) (result i32)
    local.get 0)
  (table (;0;) 2 funcref)
  (memory (;0;) 0 65536)
  (global (;0;) i32 (i32.const 10)))
*/

// CHECK: 1 wasm-translate - Parsed functions
// CHECK-NEXT: 0 wasm-translate - Parsed globals
// CHECK-NEXT: 1 wasm-translate - Parsed memories
// CHECK-NEXT: 1 wasm-translate - Parsed tables
