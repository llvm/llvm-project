; RUN: opt -S -basic-aa -slp-vectorizer < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

; Holding a value live over a call boundary may require
; spills and fills. This is the case for <2 x double>,
; as it occupies a Q register of which there are no
; callee-saves.
 
; CHECK: load double
; CHECK: load double
; CHECK: call void @g
; CHECK: store double
; CHECK: store double
define void @f(double* %p, double* %q) {
  %addr2 = getelementptr double, double* %q, i32 1
  %addr = getelementptr double, double* %p, i32 1
  %x = load double, double* %p
  %y = load double, double* %addr
  call void @g()
  store double %x, double* %q
  store double %y, double* %addr2
  ret void
}
declare void @g()

; Check we deal with loops correctly.
;
; CHECK: store <2 x double>
; CHECK: load <2 x double>
define void @f2(double* %p, double* %q) {
entry:
  br label %loop

loop:
  %p1 = phi double [0.0, %entry], [%x, %loop]
  %p2 = phi double [0.0, %entry], [%y, %loop]
  %addr2 = getelementptr double, double* %q, i32 1
  %addr = getelementptr double, double* %p, i32 1
  store double %p1, double* %q
  store double %p2, double* %addr2

  %x = load double, double* %p
  %y = load double, double* %addr
  br label %loop
}
