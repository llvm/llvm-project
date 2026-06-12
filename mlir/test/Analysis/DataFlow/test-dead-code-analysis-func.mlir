// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(test-dead-code-analysis))" 2>&1 %s | FileCheck %s

// Test that when dead code analysis is run on a single function, we correctly
// identify that we do not know all of the predecessors.
// CHECK:      foo:
// CHECK-NEXT:   region #0
// CHECK-NEXT:     ^bb0 = live
// CHECK-NEXT: op_preds: predecessors: (none)
func.func @foo(%arg0: i32) -> i32
    attributes {tag = "foo"} {
  return {a} %arg0 : i32
}
