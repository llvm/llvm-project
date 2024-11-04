; Test floating-point conversion to/from 128-bit integers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test signed i128->f128.
define fp128 @f1(i128 %i) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, __floattitf@PLT
; CHECK: br %r14
  %conv = sitofp i128 %i to fp128
  ret fp128 %conv
}

; Test signed i128->f64.
define double @f2(i128 %i) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __floattidf@PLT
; CHECK: br %r14
  %conv = sitofp i128 %i to double
  ret double %conv
}

; Test signed i128->f32.
define float @f3(i128 %i) {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, __floattisf@PLT
; CHECK: br %r14
  %conv = sitofp i128 %i to float
  ret float %conv
}

; Test unsigned i128->f128.
define fp128 @f4(i128 %i) {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, __floatuntitf@PLT
; CHECK: br %r14
  %conv = uitofp i128 %i to fp128
  ret fp128 %conv
}

; Test unsigned i128->f64.
define double @f5(i128 %i) {
; CHECK-LABEL: f5:
; CHECK: brasl %r14, __floatuntidf@PLT
; CHECK: br %r14
  %conv = uitofp i128 %i to double
  ret double %conv
}

; Test unsigned i128->f32.
define float @f6(i128 %i) {
; CHECK-LABEL: f6:
; CHECK: brasl %r14, __floatuntisf@PLT
; CHECK: br %r14
  %conv = uitofp i128 %i to float
  ret float %conv
}

; Test signed f128->i128.
define i128 @f7(fp128 %f) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, __fixtfti@PLT
; CHECK: br %r14
  %conv = fptosi fp128 %f to i128
  ret i128 %conv
}

; Test signed f64->i128.
define i128 @f8(double %f) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, __fixdfti@PLT
; CHECK: br %r14
  %conv = fptosi double %f to i128
  ret i128 %conv
}

; Test signed f9->i128.
define i128 @f9(float %f) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, __fixsfti@PLT
; CHECK: br %r14
  %conv = fptosi float %f to i128
  ret i128 %conv
}

; Test unsigned f128->i128.
define i128 @f10(fp128 %f) {
; CHECK-LABEL: f10:
; CHECK: brasl %r14, __fixunstfti@PLT
; CHECK: br %r14
  %conv = fptoui fp128 %f to i128
  ret i128 %conv
}

; Test unsigned f64->i128.
define i128 @f11(double %f) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, __fixunsdfti@PLT
; CHECK: br %r14
  %conv = fptoui double %f to i128
  ret i128 %conv
}

; Test unsigned f32->i128.
define i128 @f12(float %f) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, __fixunssfti@PLT
; CHECK: br %r14
  %conv = fptoui float %f to i128
  ret i128 %conv
}
