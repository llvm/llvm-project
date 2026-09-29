// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

// -----
// Indexes should be inlineable
func.func @func(%arg0 : index) -> index {
  %1 = index.constant 2
  return %1 : index
}

// CHECK-LABEL: @inline_interface
func.func @inline_interface(%arg0 : index) -> index {
  // CHECK: index.constant
  // CHECK-NOT: call
  %res = call @func(%arg0) : (index) -> (index)
  // CHECK: return %[[R]]
  return %res : index
}

