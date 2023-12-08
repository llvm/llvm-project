; Test 32-bit signed comparison in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparisons with 0.
define double @f1(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f1:
; CHECK: cijl %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 0
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check comparisons with 2.
define double @f2(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f2:
; CHECK: cijl %r2, 2
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 2
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CIJ range.
define double @f3(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f3:
; CHECK: cijl %r2, 127
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 127
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which must use CHI instead.
define double @f4(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f4:
; CHECK: chi %r2, 128
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 128
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the CHI range.
define double @f5(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f5:
; CHECK: chi %r2, 32767
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 32767
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which must use CFI.
define double @f6(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f6:
; CHECK: cfi %r2, 32768
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 32768
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the signed 32-bit range.
define double @f7(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f7:
; CHECK: cfi %r2, 2147483647
; CHECK-NEXT: je
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, 2147483647
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value up, which should be treated as a negative value.
define double @f8(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f8:
; CHECK: cfi %r2, -2147483648
; CHECK-NEXT: je
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, 2147483648
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the high end of the negative CIJ range.
define double @f9(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f9:
; CHECK: cijl %r2, -1
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the CIJ range.
define double @f10(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f10:
; CHECK: cijl %r2, -128
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -128
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which must use CHI instead.
define double @f11(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f11:
; CHECK: chi %r2, -129
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -129
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the CHI range.
define double @f12(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f12:
; CHECK: chi %r2, -32768
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -32768
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which must use CFI instead.
define double @f13(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f13:
; CHECK: cfi %r2, -32769
; CHECK-NEXT: jl
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -32769
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the low end of the signed 32-bit range.
define double @f14(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f14:
; CHECK: cfi %r2, -2147483648
; CHECK-NEXT: je
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, -2147483648
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check the next value down, which should be treated as a positive value.
define double @f15(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f15:
; CHECK: cfi %r2, 2147483647
; CHECK-NEXT: je
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, -2147483649
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check that < 1 becomes <= 0.
define double @f16(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f16:
; CHECK: cijle %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check that >= 1 becomes > 0.
define double @f17(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f17:
; CHECK: cijh %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp sge i32 %i1, 1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check that > -1 becomes >= 0.
define double @f18(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f18:
; CHECK: cijhe %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp sgt i32 %i1, -1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}

; Check that <= -1 becomes < 0.
define double @f19(double %a, double %b, i32 %i1) {
; CHECK-LABEL: f19:
; CHECK: cijl %r2, 0
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp sle i32 %i1, -1
  %tmp = select i1 %cond, double %a, double %b
  %res = fadd double %tmp, 1.0
  ret double %res
}
