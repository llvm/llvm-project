// RUN: mlir-opt %s -llvm-request-c-wrappers | FileCheck %s

// CHECK: func.func private @foo() attributes {llvm.emit_c_interface}
func.func private @foo()

// CHECK: func.func @bar() attributes {llvm.emit_c_interface}
func.func @bar() {
  return
}
