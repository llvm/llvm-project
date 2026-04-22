; REQUIRES: mips-registered-target
; RUN: llc < %s -mtriple=mips -mcpu=mips32r2 | FileCheck %s

;;; f32 arithmetic

define float @fadd_f32(float %a, float %b) {
; CHECK-LABEL: fadd_f32:
; CHECK: add.s
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f32(float %a, float %b) {
; CHECK-LABEL: fsub_f32:
; CHECK: sub.s
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f32(float %a, float %b) {
; CHECK-LABEL: fmul_f32:
; CHECK: mul.s
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

define float @fdiv_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_f32:
; CHECK: div.s
  %r = call float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

define float @fneg_f32(float %a) {
; CHECK-LABEL: fneg_f32:
; CHECK: neg.s
  %r = call float @llvm.fneg.f32(float %a)
  ret float %r
}

;;; Conversions

define float @sitofp_i32_f32(i32 %a) {
; CHECK-LABEL: sitofp_i32_f32:
; CHECK: cvt.s.w
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

define i32 @fptosi_f32_i32(float %a) {
; CHECK-LABEL: fptosi_f32_i32:
; CHECK: trunc.w.s
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

;;; Compare

define i1 @fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: fcmp_oeq:
; CHECK: c.eq.s
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  ret i1 %r
}

;;; Fast-math flags

; fast on fadd.f32 — same add.s instruction on MIPS
define float @fadd_fast_f32(float %a, float %b) {
; CHECK-LABEL: fadd_fast_f32:
; CHECK: add.s
  %r = call fast float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; nnan nsz on fmul.f32 — same mul.s instruction on MIPS
define float @fmul_nnan_nsz_f32(float %a, float %b) {
; CHECK-LABEL: fmul_nnan_nsz_f32:
; CHECK: mul.s
  %r = call nnan nsz float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

; reassoc on fdiv.f32 — same div.s instruction on MIPS
define float @fdiv_reassoc_f32(float %a, float %b) {
; CHECK-LABEL: fdiv_reassoc_f32:
; CHECK: div.s
  %r = call reassoc float @llvm.fdiv.f32(float %a, float %b)
  ret float %r
}

; contract on fmul+fadd — mips32r2 has no FMA for f32, produces separate mul.s + add.s
define float @fmadd_contract_f32(float %a, float %b, float %c) {
; CHECK-LABEL: fmadd_contract_f32:
; CHECK: mul.s
; CHECK: add.s
  %mul = call contract float @llvm.fmul.f32(float %a, float %b)
  %add = call contract float @llvm.fadd.f32(float %mul, float %c)
  ret float %add
}

declare float @llvm.fadd.f32(float, float)
declare float @llvm.fsub.f32(float, float)
declare float @llvm.fmul.f32(float, float)
declare float @llvm.fdiv.f32(float, float)
declare float @llvm.fneg.f32(float)
declare float @llvm.sitofp.f32.i32(i32)
declare i32 @llvm.fptosi.i32.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)
