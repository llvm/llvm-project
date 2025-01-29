// Test filtering by "before"
// RUN: rm -rf %t || true
// RUN: mlir-opt %s -mlir-print-ir-tree-dir=%t \
// RUN:   -pass-pipeline='builtin.module(builtin.module(func.func(cse,canonicalize)))' \
// RUN:   -mlir-print-ir-before=cse
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symB/0_0_0_cse.mlir
// RUN: test ! -f %t/builtin_module_outer/builtin_module_inner/func_func_symB/0_0_1_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symC/0_0_0_cse.mlir
// RUN: test ! -f %t/builtin_module_outer/builtin_module_inner/func_func_symC/0_0_1_canonicalize.mlir

// Test printing after all and the counter mechanism.
// RUN: rm -rf %t || true
// RUN: mlir-opt %s -mlir-print-ir-tree-dir=%t \
// RUN:   -pass-pipeline='builtin.module(canonicalize,canonicalize,func.func(cse),builtin.module(canonicalize,func.func(cse,canonicalize),cse),cse)' \
// RUN:   -mlir-print-ir-after-all
// RUN: test -f %t/builtin_module_outer/0_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/1_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/func_func_symA/1_0_cse.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/1_0_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symB/1_0_0_cse.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symB/1_0_1_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symC/1_0_0_cse.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/func_func_symC/1_0_1_canonicalize.mlir
// RUN: test -f %t/builtin_module_outer/builtin_module_inner/1_1_cse.mlir
// RUN: test -f %t/builtin_module_outer/2_cse.mlir

builtin.module @outer {

  func.func @symA() {
    return
  }

  builtin.module @inner {
    func.func @symB() {
      return
    }
    func.func @symC() {
      return
    }
  }
}
