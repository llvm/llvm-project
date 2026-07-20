// RUN: mlir-opt %s --wrap-func-in-module | FileCheck %s

func.func @foo() {
  return
}

func.func @bar() {
  return
}

// CHECK:      module {
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @foo() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @bar() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
