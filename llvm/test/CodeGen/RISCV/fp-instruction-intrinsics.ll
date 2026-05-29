; REQUIRES: riscv-registered-target
; RUN: llc < %s -mtriple=riscv64 -mattr=+d -verify-machineinstrs -target-abi=lp64d | FileCheck %s

;;; f64 arithmetic

define double @fadd_f64(double %a, double %b) {
; CHECK-LABEL: fadd_f64:
; CHECK: fadd.d fa0, fa0, fa1
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

define double @fsub_f64(double %a, double %b) {
; CHECK-LABEL: fsub_f64:
; CHECK: fsub.d fa0, fa0, fa1
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

define double @fmul_f64(double %a, double %b) {
; CHECK-LABEL: fmul_f64:
; CHECK: fmul.d fa0, fa0, fa1
  %r = call double @llvm.fmul.f64(double %a, double %b)
  ret double %r
}

define double @fdiv_f64(double %a, double %b) {
; CHECK-LABEL: fdiv_f64:
; CHECK: fdiv.d fa0, fa0, fa1
  %r = call double @llvm.fdiv.f64(double %a, double %b)
  ret double %r
}

;;; f32 arithmetic

define float @fadd_f32(float %a, float %b) {
; CHECK-LABEL: fadd_f32:
; CHECK: fadd.s fa0, fa0, fa1
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f32(float %a, float %b) {
; CHECK-LABEL: fsub_f32:
; CHECK: fsub.s fa0, fa0, fa1
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f32(float %a, float %b) {
; CHECK-LABEL: fmul_f32:
; CHECK: fmul.s fa0, fa0, fa1
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

define float @fdiv_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_f32:
; CHECK: fdiv.s fa0, fa0, fa1
  %r = call float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

;;; Conversions

define float @fptrunc(double %a) {
; CHECK-LABEL: fptrunc:
; CHECK: fcvt.s.d fa0, fa0
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  ret float %r
}

define double @fpext(float %a) {
; CHECK-LABEL: fpext:
; CHECK: fcvt.d.s fa0, fa0
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

define float @sitofp_i32_f32(i32 %a) {
; CHECK-LABEL: sitofp_i32_f32:
; CHECK: fcvt.s.w fa0, a0
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

define float @uitofp_i32_f32(i32 %a) {
; CHECK-LABEL: uitofp_i32_f32:
; CHECK: fcvt.s.wu fa0, a0
  %r = call float @llvm.uitofp.f32.i32(i32 %a)
  ret float %r
}

define i32 @fptosi_f32_i32(float %a) {
; CHECK-LABEL: fptosi_f32_i32:
; CHECK: fcvt.w.s a0, fa0
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

define i32 @fptoui_f32_i32(float %a) {
; CHECK-LABEL: fptoui_f32_i32:
; CHECK: fcvt.wu.s a0, fa0
  %r = call i32 @llvm.fptoui.i32.f32(float %a)
  ret i32 %r
}

;;; Compare

define i1 @fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: fcmp_oeq:
; CHECK: feq.s a0, fa0, fa1
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  ret i1 %r
}

;;; Fast-math flags

; fast on fadd.f32 -- same fadd.s instruction on RISCV
define float @fadd_fast_f32(float %a, float %b) {
; CHECK-LABEL: fadd_fast_f32:
; CHECK: fadd.s fa0, fa0, fa1
  %r = call fast float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; nnan nsz on fmul.f32 -- same fmul.s instruction on RISCV
define float @fmul_nnan_nsz_f32(float %a, float %b) {
; CHECK-LABEL: fmul_nnan_nsz_f32:
; CHECK: fmul.s fa0, fa0, fa1
  %r = call nnan nsz float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

; reassoc on fdiv.f32 -- same fdiv.s instruction on RISCV
define float @fdiv_reassoc_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_reassoc_f32:
; CHECK: fdiv.s fa0, fa0, fa1
  %r = call reassoc float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

; contract on fmul+fadd -> fmadd.s (FMA contraction)
define float @fmadd_contract_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fmadd_contract_f32:
; CHECK: fmadd.s fa0, fa0, fa1, fa2
  %mul = call contract float @llvm.fmul.f32(float %a, float %b)
  %add = call contract float @llvm.fadd.f32(float %mul, float %c)
  ret float %add
}

; contract on fmul+fadd for f64 -> fmadd.d
define double @fmadd_contract_f64(double %a, double %b, double %c) {
; CHECK-LABEL: fmadd_contract_f64:
; CHECK: fmadd.d fa0, fa0, fa1, fa2
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
