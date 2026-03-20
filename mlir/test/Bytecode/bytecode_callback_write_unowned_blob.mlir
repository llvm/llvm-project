// RUN: mlir-opt %s -split-input-file --test-bytecode-roundtrip="test-kind=7" | FileCheck %s

func.func @base_test(%arg0: !test.i32) {
  return
}

// CHECK: Writing unowned blob...
// CHECK: Successfully read the unowned blob.
// CHECK: func.func @base_test([[ARG0:%.+]]: !test.i32) {
