; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone -aarch64-neon-syntax=apple | FileCheck %s
; rdar://10263824

define i1 @fcmp_float1(float %a) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_float1
; CHECK: fcmp s0, #0.0
; CHECK: cset w0, ne
  %cmp = fcmp une float %a, 0.000000e+00
  ret i1 %cmp
}

define i1 @fcmp_float2(float %a, float %b) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_float2
; CHECK: fcmp s0, s1
; CHECK: cset w0, ne
  %cmp = fcmp une float %a, %b
  ret i1 %cmp
}

define i1 @fcmp_double1(double %a) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_double1
; CHECK: fcmp d0, #0.0
; CHECK: cset w0, ne
  %cmp = fcmp une double %a, 0.000000e+00
  ret i1 %cmp
}

define i1 @fcmp_double2(double %a, double %b) nounwind ssp {
entry:
; CHECK-LABEL: @fcmp_double2
; CHECK: fcmp d0, d1
; CHECK: cset w0, ne
  %cmp = fcmp une double %a, %b
  ret i1 %cmp
}

; Check each fcmp condition
define float @fcmp_oeq(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_oeq
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], eq

  %cmp = fcmp oeq float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ogt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ogt
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], gt

  %cmp = fcmp ogt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_oge(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_oge
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], ge

  %cmp = fcmp oge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_olt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_olt
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], mi

  %cmp = fcmp olt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ole(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ole
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], ls

  %cmp = fcmp ole float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ord(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ord
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], vc
  %cmp = fcmp ord float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uno(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_uno
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], vs
  %cmp = fcmp uno float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ugt(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ugt
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], hi
  %cmp = fcmp ugt float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_uge(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_uge
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], pl
  %cmp = fcmp uge float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ult(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ult
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], lt
  %cmp = fcmp ult float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_ule(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ule
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], le
  %cmp = fcmp ule float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

define float @fcmp_une(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_une
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel s0, s[[ONE]], s[[ZERO]], ne
  %cmp = fcmp une float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

; Possible opportunity for improvement.  See comment in
; ARM64TargetLowering::LowerSETCC()
define float @fcmp_one(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_one
;	fcmp	s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel [[TMP:s[0-9]+]], s[[ONE]], s[[ZERO]], mi
; CHECK: fcsel s0, s[[ONE]], [[TMP]], gt
  %cmp = fcmp one float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}

; Possible opportunity for improvement.  See comment in
; ARM64TargetLowering::LowerSETCC()
define float @fcmp_ueq(float %a, float %b) nounwind ssp {
; CHECK-LABEL: @fcmp_ueq
; CHECK: fcmp s0, s1
; CHECK-DAG: movi.2d v[[ZERO:[0-9]+]], #0
; CHECK-DAG: fmov s[[ONE:[0-9]+]], #1.0
; CHECK: fcsel [[TMP:s[0-9]+]], s[[ONE]], s[[ZERO]], eq
; CHECK: fcsel s0, s[[ONE]], [[TMP]], vs
  %cmp = fcmp ueq float %a, %b
  %conv = uitofp i1 %cmp to float
  ret float %conv
}
