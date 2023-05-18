// RUN: mlir-opt -inline %s | FileCheck %s

// This could crash the inliner, make sure it does not.

func.func @A() {
  call @B() { inA } : () -> ()
  return
}

func.func @B() {
  call @E() : () -> ()
  return
}

func.func @C() {
  call @D() : () -> ()
  return
}

func.func private @D() {
  call @B() { inD } : () -> ()
  return
}

func.func @E() {
  call @fabsf() : () -> ()
  return
}

func.func private @fabsf()

// CHECK: func @A() {
// CHECK:   call @fabsf() : () -> ()
// CHECK:   return
// CHECK: }
// CHECK: func @B() {
// CHECK:   call @fabsf() : () -> ()
// CHECK:   return
// CHECK: }
// CHECK: func @C() {
// CHECK:   call @fabsf() : () -> ()
// CHECK:   return
// CHECK: }
// CHECK: func @E() {
// CHECK:   call @fabsf() : () -> ()
// CHECK:   return
// CHECK: }
// CHECK: func private @fabsf()
