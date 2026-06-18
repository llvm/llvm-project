// RUN: yaml2obj %S/inputs/import.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
(import "my_module" "foo" (func $foo (param i32)))
(import "my_module" "bar" (func $bar (param i32)))
(import "my_module" "table" (table $round 2 funcref))
(import "my_module" "mem" (memory $mymem 2))
(import "my_module" "glob" (global $globglob i32))
(import "my_other_module" "glob_mut" (global $glob_mut (mut i32)))
)
*/

// CHECK-LABEL:   wasmssa.import_func "foo" from "my_module" as @func_0 {type = (i32) -> ()}
// CHECK:         wasmssa.import_func "bar" from "my_module" as @func_1 {type = (i32) -> ()}
// CHECK:         wasmssa.import_table "table" from "my_module" as @table_0 {type = !wasmssa<tabletype !wasmssa.funcref [2:]>}
// CHECK:         wasmssa.import_mem "mem" from "my_module" as @mem_0 {limits = !wasmssa<limit[2:]>}
// CHECK:         wasmssa.import_global "glob" from "my_module" as @global_0 : i32
// CHECK:         wasmssa.import_global "glob_mut" from "my_other_module" as @global_1 mutable : i32
