; REQUIRES: hexagon-registered-target
; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

;;; f64 arithmetic (hexagonv66 native ops)

define double @fadd_f64(double %a, double %b) #0 {
; CHECK-LABEL: fadd_f64:
; CHECK: dfadd
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

define double @fsub_f64(double %a, double %b) #0 {
; CHECK-LABEL: fsub_f64:
; CHECK: dfsub
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

;;; f32 arithmetic

define float @fadd_f32(float %a, float %b) #0 {
; CHECK-LABEL: fadd_f32:
; CHECK: sfadd
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f32(float %a, float %b) #0 {
; CHECK-LABEL: fsub_f32:
; CHECK: sfsub
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f32(float %a, float %b) #0 {
; CHECK-LABEL: fmul_f32:
; CHECK: sfmpy
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

;;; Conversions

define float @fptrunc(double %a) #0 {
; CHECK-LABEL: fptrunc:
; CHECK: convert_df2sf
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  ret float %r
}

define double @fpext(float %a) #0 {
; CHECK-LABEL: fpext:
; CHECK: convert_sf2df
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

define float @sitofp_i32_f32(i32 %a) #0 {
; CHECK-LABEL: sitofp_i32_f32:
; CHECK: convert_w2sf
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

define float @uitofp_i32_f32(i32 %a) #0 {
; CHECK-LABEL: uitofp_i32_f32:
; CHECK: convert_uw2sf
  %r = call float @llvm.uitofp.f32.i32(i32 %a)
  ret float %r
}

define i32 @fptosi_f32_i32(float %a) #0 {
; CHECK-LABEL: fptosi_f32_i32:
; CHECK: convert_sf2w
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

;;; Compare

define i32 @fcmp_oeq(float %a, float %b) #0 {
; CHECK-LABEL: fcmp_oeq:
; CHECK: sfcmp.eq
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  %ext = zext i1 %r to i32
  ret i32 %ext
}

;;; Fast-math flags

; fast on fadd.f32 -- same sfadd instruction on Hexagon
define float @fadd_fast_f32(float %a, float %b) #0 {
; CHECK-LABEL: fadd_fast_f32:
; CHECK: sfadd
  %r = call fast float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

; nnan nsz on fmul.f32 -- same sfmpy instruction on Hexagon
define float @fmul_nnan_nsz_f32(float %a, float %b) #0 {
; CHECK-LABEL: fmul_nnan_nsz_f32:
; CHECK: sfmpy
  %r = call nnan nsz float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

; contract on fmul+fadd -> FMA accumulate: rx += sfmpy(ry, rz)
define float @fmadd_contract_f32(float %a, float %b, float %c) #0 {
; CHECK-LABEL: fmadd_contract_f32:
; CHECK: += sfmpy(
  %mul = call contract float @llvm.fmul.f32(float %a, float %b)
  %add = call contract float @llvm.fadd.f32(float %mul, float %c)
  ret float %add
}

declare double @llvm.fadd.f64(double, double)
declare double @llvm.fsub.f64(double, double)
declare float @llvm.fadd.f32(float, float)
declare float @llvm.fsub.f32(float, float)
declare float @llvm.fmul.f32(float, float)
declare float @llvm.fptrunc.f32.f64(double)
declare double @llvm.fpext.f64.f32(float)
declare float @llvm.sitofp.f32.i32(i32)
declare float @llvm.uitofp.f32.i32(i32)
declare i32 @llvm.fptosi.i32.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)

attributes #0 = { nounwind "target-cpu"="hexagonv66" }
