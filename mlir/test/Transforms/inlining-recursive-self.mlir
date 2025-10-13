// RUN: mlir-opt %s -inline='default-pipeline=' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='default-pipeline=' | FileCheck %s

// CHECK-LABEL: func.func @b0
func.func @b0() {
  // CHECK:         call @b0
  // CHECK-NEXT:    call @b1
  // CHECK-NEXT:    call @b0
  // CHECK-NEXT:    call @b1
  // CHECK-NEXT:    call @b0
  func.call @b0() : () -> ()
  func.call @b1() : () -> ()
  func.call @b0() : () -> ()
  func.call @b1() : () -> ()
  func.call @b0() : () -> ()
  return
}
func.func @b1() {
  func.call @b1() : () -> ()
  func.call @b1() : () -> ()
  return
}
