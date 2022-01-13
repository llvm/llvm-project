// RUN: mlir-opt -inline %s | FileCheck %s

// This could crash the inliner, make sure it does not.

func @A() {
  call @B() { inA } : () -> ()
  return
}

func @B() {
  call @E() : () -> ()
  return
}

func @C() {
  call @D() : () -> ()
  return
}

func private @D() {
  call @B() { inD } : () -> ()
  return
}

func @E() {
  call @fabsf() : () -> ()
  return
}

func private @fabsf()

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
