// RUN: not mlir-opt -split-input-file %s -mlir-print-op-on-diagnostic 2>&1 | FileCheck %s

// This file tests the functionality of 'mlir-print-op-on-diagnostic'.

// CHECK: {{invalid to use 'test.invalid_attr'}}
// CHECK: see current operation:
// CHECK-NEXT: "builtin.module"()
module attributes {test.invalid_attr} {}

// -----

func.func @foo() {
  "test.foo"(%cst) : (index) -> ()
  // CHECK: {{operand #0 does not dominate this use}}
  // CHECK: {{see current operation: "test.foo"(.*)}}
  %cst = arith.constant 0 : index
  return
}
