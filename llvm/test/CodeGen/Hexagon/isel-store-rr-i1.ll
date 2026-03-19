; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon-unknown-linux-gnu"

define i32 @f0(float %a0, double %a1, i1 %a2, i16 %a3, i8 %a4) {
; CHECK-LABEL: f0:
; CHECK:     memb(r1+r0<<#2) = r2
b0:
  %v0 = alloca double, align 8
  %v1 = load i32, ptr poison, align 4
  %v2 = or i32 42, %v1
  %v3 = getelementptr ptr, ptr %v0, i32 %v2
  store i1 false, ptr %v3, align 1
  ret i32 %v2
}
