; Test 64-bit inequality comparisons in which the second operand is a constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparisons with 0.
define double @f1(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f1:
; CHECK: cgijlh %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 0
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CGIJ range.
define double @f2(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f2:
; CHECK: cgijlh %r2, 127
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 127
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which must use CGHI instead.
define double @f3(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f3:
; CHECK: cghi %r2, 128
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 128
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CGHI range.
define double @f4(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f4:
; CHECK: cghi %r2, 32767
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 32767
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which must use CGFI.
define double @f5(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f5:
; CHECK: cgfi %r2, 32768
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 32768
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CGFI range.
define double @f6(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f6:
; CHECK: cgfi %r2, 2147483647
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 2147483647
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which should use CLGFI instead.
define double @f7(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f7:
; CHECK: clgfi %r2, 2147483648
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 2147483648
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CLGFI range.
define double @f8(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f8:
; CHECK: clgfi %r2, 4294967295
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 4294967295
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which must use a register comparison.
define double @f9(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f9:
; CHECK: cgrjlh %r2,
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, 4294967296
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the negative CGIJ range.
define double @f10(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f10:
; CHECK: cgijlh %r2, -1
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the CGIJ range.
define double @f11(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f11:
; CHECK: cgijlh %r2, -128
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -128
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which must use CGHI instead.
define double @f12(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f12:
; CHECK: cghi %r2, -129
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -129
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the CGHI range.
define double @f13(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f13:
; CHECK: cghi %r2, -32768
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -32768
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which must use CGFI instead.
define double @f14(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f14:
; CHECK: cgfi %r2, -32769
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -32769
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the CGFI range.
define double @f15(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f15:
; CHECK: cgfi %r2, -2147483648
; CHECK-NEXT: jlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -2147483648
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which must use register comparison.
define double @f16(double %a, double %b, i64 %i1) {
; CHECK-LABEL: f16:
; CHECK: cgrjlh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ne i64 %i1, -2147483649
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}
