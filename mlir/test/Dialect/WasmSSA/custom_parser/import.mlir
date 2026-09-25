// RUN: mlir-opt %s | FileCheck %s

module {
  wasmssa.import_func "foo" from "my_module" as @func_0 : (i32) -> () {sym_visibility = "nested"}
  wasmssa.import_func "bar" from "my_module" as @func_1 : (i32) -> () {sym_visibility = "nested"}
  wasmssa.import_table "table" from "my_module" as @table_0 !wasmssa<tabletype !wasmssa.funcref [2:]> {sym_visibility = "nested"}
  wasmssa.import_mem "mem" from "my_module" as @mem_0 !wasmssa<limit[2:]> {sym_visibility = "nested"}
  wasmssa.import_global "glob" from "my_module" as @global_0 : i32
  wasmssa.import_global "glob_mut" from "my_other_module" as @global_1 mutable : i32
}

// CHECK-LABEL:   wasmssa.import_func "foo" from "my_module" as @func_0 : (i32) -> () {sym_visibility = "nested"}
// CHECK:         wasmssa.import_func "bar" from "my_module" as @func_1 : (i32) -> () {sym_visibility = "nested"}
// CHECK:         wasmssa.import_table "table" from "my_module" as @table_0 !wasmssa<tabletype !wasmssa.funcref [2:]> {sym_visibility = "nested"}
// CHECK:         wasmssa.import_mem "mem" from "my_module" as @mem_0 !wasmssa<limit[2:]> {sym_visibility = "nested"}
// CHECK:         wasmssa.import_global "glob" from "my_module" as @global_0 : i32
// CHECK:         wasmssa.import_global "glob_mut" from "my_other_module" as @global_1 mutable : i32
