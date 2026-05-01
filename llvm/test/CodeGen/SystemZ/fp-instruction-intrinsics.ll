; REQUIRES: systemz-registered-target
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

;;; f64 arithmetic

define double @fadd_f64(double %a, double %b) {
; CHECK-LABEL: fadd_f64:
; CHECK: adbr
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

define double @fsub_f64(double %a, double %b) {
; CHECK-LABEL: fsub_f64:
; CHECK: sdbr
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

define double @fmul_f64(double %a, double %b) {
; CHECK-LABEL: fmul_f64:
; CHECK: mdbr
  %r = call double @llvm.fmul.f64(double %a, double %b)
  ret double %r
}

define double @fdiv_f64(double %a, double %b) {
; CHECK-LABEL: fdiv_f64:
; CHECK: ddbr
  %r = call double @llvm.fdiv.f64(double %a, double %b)
  ret double %r
}

;;; f32 arithmetic

define float @fadd_f32(float %a, float %b) {
; CHECK-LABEL: fadd_f32:
; CHECK: aebr
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f32(float %a, float %b) {
; CHECK-LABEL: fsub_f32:
; CHECK: sebr
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f32(float %a, float %b) {
; CHECK-LABEL: fmul_f32:
; CHECK: meebr
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

define float @fdiv_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_f32:
; CHECK: debr
  %r = call float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

;;; Conversions

define float @fptrunc(double %a) {
; CHECK-LABEL: fptrunc:
; CHECK: ledbr
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  ret float %r
}

define double @fpext(float %a) {
; CHECK-LABEL: fpext:
; CHECK: ldebr
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

define float @sitofp_i32_f32(i32 %a) {
; CHECK-LABEL: sitofp_i32_f32:
; CHECK: cefbr
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

define float @uitofp_i32_f32(i32 %a) {
; CHECK-LABEL: uitofp_i32_f32:
; CHECK: celfbr
  %r = call float @llvm.uitofp.f32.i32(i32 %a)
  ret float %r
}

define i32 @fptosi_f32_i32(float %a) {
; CHECK-LABEL: fptosi_f32_i32:
; CHECK: cfebr
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

define i32 @fptoui_f32_i32(float %a) {
; CHECK-LABEL: fptoui_f32_i32:
; CHECK: clfebr
  %r = call i32 @llvm.fptoui.i32.f32(float %a)
  ret i32 %r
}

;;; Compare

define i1 @fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: fcmp_oeq:
; CHECK: cebr
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  ret i1 %r
}

;;; Fast-math flags

; fast on fadd.f32 -- same aebr instruction on SystemZ
define float @fadd_fast_f32(float %a, float %b) {
; CHECK-LABEL: fadd_fast_f32:
; CHECK: aebr
  %r = call fast float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; nnan nsz on fmul.f32 -- same meebr instruction on SystemZ
define float @fmul_nnan_nsz_f32(float %a, float %b) {
; CHECK-LABEL: fmul_nnan_nsz_f32:
; CHECK: meebr
  %r = call nnan nsz float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

; reassoc on fdiv.f32 -- same debr instruction on SystemZ
define float @fdiv_reassoc_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_reassoc_f32:
; CHECK: debr
  %r = call reassoc float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

; contract on fmul+fadd -> wfmasb (FMA contraction, f32, z14 vector FP)
define float @fmadd_contract_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fmadd_contract_f32:
; CHECK: wfmasb
  %mul = call contract float @llvm.fmul.f32(float %a, float %b)
  %add = call contract float @llvm.fadd.f32(float %mul, float %c)
  ret float %add
}

; contract on fmul+fadd for f64 -> wfmadb (z14 vector FP)
define double @fmadd_contract_f64(double %a, double %b, double %c) {
; CHECK-LABEL: fmadd_contract_f64:
; CHECK: wfmadb
  %mul = call contract double @llvm.fmul.f64(double %a, double %b)
  %add = call contract double @llvm.fadd.f64(double %mul, double %c)
  ret double %add
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
