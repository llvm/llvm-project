; Test conversion of floating-point values to unsigned integers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test f16->i32.
define i32 @f0(half %f) {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT: clfebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui half %f to i32
  ret i32 %conv
}

; Test f32->i32.
define i32 @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: clfebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui float %f to i32
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: clfdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui double %f to i32
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clfxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = fptoui fp128 %f to i32
  ret i32 %conv
}

; Test f16->i64.
define i64 @f4(half %f) {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT: clgebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui half %f to i64
  ret i64 %conv
}

; Test f32->i64.
define i64 @f5(float %f) {
; CHECK-LABEL: f5:
; CHECK: clgebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui float %f to i64
  ret i64 %conv
}

; Test f64->i64.
define i64 @f6(double %f) {
; CHECK-LABEL: f6:
; CHECK: clgdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = fptoui double %f to i64
  ret i64 %conv
}

; Test f128->i64.
define i64 @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clgxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = fptoui fp128 %f to i64
  ret i64 %conv
}
