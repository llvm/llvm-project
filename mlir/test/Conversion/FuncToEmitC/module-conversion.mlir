// RUN: mlir-opt -split-input-file -convert-func-to-emitc %s | FileCheck %s

module @outer {
  module @inner {
  }
}

// CHECK: module @outer {
// CHECK-NEXT: module @inner {
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

module @outer {
  module @inner {
    func.func @func_in_inner() {
      return
    }
  }
}

// CHECK:      module @outer {
// CHECK-NEXT:   emitc.class @inner {
// CHECK-NEXT:     emitc.func @func_in_inner() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
