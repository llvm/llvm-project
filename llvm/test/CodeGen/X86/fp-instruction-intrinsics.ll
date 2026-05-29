; REQUIRES: x86-registered-target
; RUN: llc < %s -mtriple=x86_64-pc-linux -mattr=+sse2 | FileCheck %s

;;; f64 arithmetic

define double @fadd_f64(double %a, double %b) {
; CHECK-LABEL: fadd_f64:
; CHECK: addsd
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

define double @fsub_f64(double %a, double %b) {
; CHECK-LABEL: fsub_f64:
; CHECK: subsd
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

define double @fmul_f64(double %a, double %b) {
; CHECK-LABEL: fmul_f64:
; CHECK: mulsd
  %r = call double @llvm.fmul.f64(double %a, double %b)
  ret double %r
}

define double @fdiv_f64(double %a, double %b) {
; CHECK-LABEL: fdiv_f64:
; CHECK: divsd
  %r = call double @llvm.fdiv.f64(double %a, double %b)
  ret double %r
}

;;; f32 arithmetic

define float @fadd_f32(float %a, float %b) {
; CHECK-LABEL: fadd_f32:
; CHECK: addss
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f32(float %a, float %b) {
; CHECK-LABEL: fsub_f32:
; CHECK: subss
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f32(float %a, float %b) {
; CHECK-LABEL: fmul_f32:
; CHECK: mulss
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

define float @fdiv_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_f32:
; CHECK: divss
  %r = call float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

;;; Conversions

define float @fptrunc(double %a) {
; CHECK-LABEL: fptrunc:
; CHECK: cvtsd2ss
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  ret float %r
}

define double @fpext(float %a) {
; CHECK-LABEL: fpext:
; CHECK: cvtss2sd
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

define float @sitofp_i32_f32(i32 %a) {
; CHECK-LABEL: sitofp_i32_f32:
; CHECK: cvtsi2ss
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

define float @uitofp_i32_f32(i32 %a) {
; CHECK-LABEL: uitofp_i32_f32:
; CHECK: cvtsi2ss
  %r = call float @llvm.uitofp.f32.i32(i32 %a)
  ret float %r
}

define i32 @fptosi_f32_i32(float %a) {
; CHECK-LABEL: fptosi_f32_i32:
; CHECK: cvttss2si
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

define i32 @fptoui_f32_i32(float %a) {
; CHECK-LABEL: fptoui_f32_i32:
; CHECK: cvttss2si
  %r = call i32 @llvm.fptoui.i32.f32(float %a)
  ret i32 %r
}

;;; Compare

define i1 @fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: fcmp_oeq:
; CHECK: cmpeqss
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  ret i1 %r
}

;;; Fast-math flags

; nnan nsz flags -- same codegen as plain fadd on x86 SSE2
define float @fadd_nnan_nsz_f32(float %a, float %b) {
; CHECK-LABEL: fadd_nnan_nsz_f32:
; CHECK: addss
  %r = call nnan nsz float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; fast flags -- same codegen as plain fmul on x86 SSE2
define float @fmul_fast_f32(float %a, float %b) {
; CHECK-LABEL: fmul_fast_f32:
; CHECK: mulss
  %r = call fast float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

; reassoc on fdiv -- still generates divss on plain SSE2
define float @fdiv_reassoc_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_reassoc_f32:
; CHECK: divss
  %r = call reassoc float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

declare double @llvm.fadd.f64(double, double)
declare double @llvm.fsub.f64(double, double)
declare double @llvm.fmul.f64(double, double)
declare double @llvm.fdiv.f64(double, double)
declare float @llvm.fadd.f32(float, float)
declare float @llvm.fsub.f32(float, float)
declare float @llvm.fmul.f32(float, float)
declare float @llvm.fdiv.f32(float, float)
declare float @llvm.fptrunc.f32.f64(double)
declare double @llvm.fpext.f64.f32(float)
declare float @llvm.sitofp.f32.i32(i32)
declare float @llvm.uitofp.f32.i32(i32)
declare i32 @llvm.fptosi.i32.f32(float)
declare i32 @llvm.fptoui.i32.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)
