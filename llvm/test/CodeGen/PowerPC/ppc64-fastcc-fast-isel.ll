; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-vsx -fast-isel -fast-isel-abort=1 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define fastcc i64 @g1(i64 %g1, double %f1, i64 %g2, double %f2, i64 %g3, double %f3, i64 %g4, double %f4) #0 {
  ret i64 %g1

; CHECK-LABEL: @g1
; CHECK-NOT: mr 3,
; CHECK: blr
}

define fastcc i64 @g2(i64 %g1, double %f1, i64 %g2, double %f2, i64 %g3, double %f3, i64 %g4, double %f4) #0 {
  ret i64 %g2

; CHECK-LABEL: @g2
; CHECK: mr 3, 4
; CHECK-NEXT: blr
}

define fastcc i64 @g3(i64 %g1, double %f1, i64 %g2, double %f2, i64 %g3, double %f3, i64 %g4, double %f4) #0 {
  ret i64 %g3

; CHECK-LABEL: @g3
; CHECK: mr 3, 5
; CHECK-NEXT: blr
}

define fastcc double @f2(i64 %g1, double %f1, i64 %g2, double %f2, i64 %g3, double %f3, i64 %g4, double %f4) #0 {
  ret double %f2

; CHECK-LABEL: @f2
; CHECK: fmr 1, 2
; CHECK-NEXT: blr
}

define void @cg2(i64 %v) #0 {
  call fastcc i64 @g1(i64 0, double 0.0, i64 %v, double 0.0, i64 0, double 0.0, i64 0, double 0.0)
  ret void

; CHECK-LABEL: @cg2
; CHECK: mr 4, 3
; CHECK: blr
}

define void @cf2(double %v) #0 {
  call fastcc i64 @g1(i64 0, double 0.0, i64 0, double %v, i64 0, double 0.0, i64 0, double 0.0)
  ret void

; CHECK-LABEL: @cf2
; CHECK: fmr 2, 1
; CHECK: blr
}

attributes #0 = { nounwind }

