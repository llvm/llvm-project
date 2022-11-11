; Test incoming GPR, FPR and stack arguments when no extension type is given.
; This type of argument is used for passing structures, etc.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs | FileCheck %s

; Do some arithmetic so that we can see the register being used.
define i8 @f1(i8 %r2) {
; CHECK-LABEL: f1:
; CHECK: ahi %r2, 1
; CHECK: br %r14
  %y = add i8 %r2, 1
  ret i8 %y
}

define i16 @f2(i8 %r2, i16 %r3) {
; CHECK-LABEL: f2:
; CHECK: {{lr|lgr}} %r2, %r3
; CHECK: br %r14
  ret i16 %r3
}

define i32 @f3(i8 %r2, i16 %r3, i32 %r4) {
; CHECK-LABEL: f3:
; CHECK: {{lr|lgr}} %r2, %r4
; CHECK: br %r14
  ret i32 %r4
}

define i64 @f4(i8 %r2, i16 %r3, i32 %r4, i64 %r5) {
; CHECK-LABEL: f4:
; CHECK: {{lr|lgr}} %r2, %r5
; CHECK: br %r14
  ret i64 %r5
}

; Do some arithmetic so that we can see the register being used.
define float @f5(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0) {
; CHECK-LABEL: f5:
; CHECK: aebr %f0, %f0
; CHECK: br %r14
  %y = fadd float %f0, %f0
  ret float %y
}

define double @f6(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2) {
; CHECK-LABEL: f6:
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  ret double %f2
}

; fp128s are passed indirectly.  Do some arithmetic so that the value
; must be interpreted as a float, rather than as a block of memory to
; be copied.
define void @f7(ptr %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                fp128 %r6) {
; CHECK-LABEL: f7:
; CHECK: ld %f0, 0(%r6)
; CHECK: ld %f2, 8(%r6)
; CHECK: axbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %y = fadd fp128 %r6, %r6
  store fp128 %y, ptr %r2
  ret void
}

define i64 @f8(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
               fp128 %r6, i64 %s1) {
; CHECK-LABEL: f8:
; CHECK: lg %r2, 160(%r15)
; CHECK: br %r14
  ret i64 %s1
}

define float @f9(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                 fp128 %r6, i64 %s1, float %f4) {
; CHECK-LABEL: f9:
; CHECK: ler %f0, %f4
; CHECK: br %r14
  ret float %f4
}

define double @f10(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                   fp128 %r6, i64 %s1, float %f4, double %f6) {
; CHECK-LABEL: f10:
; CHECK: ldr %f0, %f6
; CHECK: br %r14
  ret double %f6
}

define i64 @f11(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                fp128 %r6, i64 %s1, float %f4, double %f6, i64 %s2) {
; CHECK-LABEL: f11:
; CHECK: lg %r2, 168(%r15)
; CHECK: br %r14
  ret i64 %s2
}

; Floats are passed right-justified.
define float @f12(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                  fp128 %r6, i64 %s1, float %f4, double %f6, i64 %s2,
                  float %s3) {
; CHECK-LABEL: f12:
; CHECK: le %f0, 180(%r15)
; CHECK: br %r14
  ret float %s3
}

; Test a case where the fp128 address is passed on the stack.
define void @f13(ptr %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                 fp128 %r6, i64 %s1, float %f4, double %f6, i64 %s2,
                 float %s3, fp128 %s4) {
; CHECK-LABEL: f13:
; CHECK: lg [[REGISTER:%r[1-5]+]], 184(%r15)
; CHECK: ld %f0, 0([[REGISTER]])
; CHECK: ld %f2, 8([[REGISTER]])
; CHECK: axbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %y = fadd fp128 %s4, %s4
  store fp128 %y, ptr %r2
  ret void
}

; Explicit fp128 return values are likewise passed indirectly.
define fp128 @f14(fp128 %r3) {
; CHECK-LABEL: f14:
; CHECK: ld %f0, 0(%r3)
; CHECK: ld %f2, 8(%r3)
; CHECK: axbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %y = fadd fp128 %r3, %r3
  ret fp128 %y
}

