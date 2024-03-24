// Test filtering by "before"
// RUN: rm -rf %t || true
// RUN: mlir-opt %s -mlir-print-ir-tree-dir=%t \
// RUN:   -pass-pipeline='builtin.module(builtin.module(func.func(cse,canonicalize)))' \
// RUN:   -mlir-print-ir-before=cse -mlir-disable-threading
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func1/0_cse.mlir
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func2/1_cse.mlir

// Test printing after all.
// RUN: rm -rf %t || true
// RUN: mlir-opt %s -mlir-print-ir-tree-dir=%t \
// RUN:   -pass-pipeline='builtin.module(builtin.module(func.func(cse,canonicalize)))' \
// RUN:   -mlir-print-ir-after-all -mlir-disable-threading
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func1/0_cse.mlir
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func1/1_canonicalize.mlir
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func2/2_cse.mlir
// RUN: test -f %t/builtin_module_top/builtin_module_middle/func_func_func2/3_canonicalize.mlir

builtin.module @top {
  builtin.module @middle {
    func.func @func1() {
      return
    }
    func.func @func2() {
      return
    }
  }
}
