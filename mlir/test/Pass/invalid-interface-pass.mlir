// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-interface-pass)' 2>&1 | FileCheck %s

// Test that we emit an error when an interface pass is added to a pass manager it can't be scheduled on.

// CHECK: unable to schedule pass '{{.*}}' on a PassManager intended to run on 'builtin.module'!

func.func @main() {
  return
}
