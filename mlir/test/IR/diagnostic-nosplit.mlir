// RUN: not mlir-opt %s -o - --split-input-file 2>&1 | FileCheck %s
// This test verifies that diagnostic handler doesn't emit splits.


// -----



func.func @constant_out_of_range() {
  // CHECK: mlir:11:8: error: 'arith.constant'
  %x = "arith.constant"() {value = 100} : () -> i1
  return
}
