// RUN: mlir-opt --pass-pipeline="builtin.module(builtin.module(test-dead-code-analysis))" %s 2>&1 | FileCheck %s

// Test that when dead code analysis runs directly on a nested module with a
// private symbol name, we account for the fact that functions within that
// module may be invoked from outside the module.
module {
  module @inner_module attributes {sym_visibility = "private"} {
    // CHECK:      nested:
    // CHECK-NEXT:  region #0
    // CHECK-NEXT:   ^bb0 = live
    // CHECK-NEXT: op_preds: predecessors: (none)
    func.func nested @nested_inner(%arg0: i32) -> i32 attributes {tag = "nested"} {
      return %arg0 : i32
    }
  }
}
