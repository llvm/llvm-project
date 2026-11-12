; Test 32-bit unsigned comparisons in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a value near the low end of the range.  We use signed forms for
; comparisons with zero, or things that are equivalent to them.
define double @f1(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f1:
; CHECK: clijh %r2, 1
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ugt i32 %i1, 1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the top of the CLIJ range.
define double @f2(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f2:
; CHECK: clijl %r2, 255
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i32 %i1, 255
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which needs a separate comparison.
define double @f3(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f3:
; CHECK: clfi %r2, 256
; CHECK: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i32 %i1, 256
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check a value near the high end of the range.
define double @f4(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f4:
; CHECK: clfi %r2, 4294967280
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i32 %i1, 4294967280
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}
