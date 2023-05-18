; RUN: llc < %s -mtriple=arm-eabi -mattr=+fullfp16 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s
; RUN: llc < %s -mtriple thumbv7a -mattr=+fullfp16 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s

; TODO: we can't pass half-precision arguments as "half" types yet. We do
; that for the time being by passing "float %f.coerce" and the necessary
; bitconverts/truncates. In these tests we pass i16 and use 1 bitconvert, which
; is the shortest way to get a half type. But when we can pass half types, we
; want to use that here.

define half @fp16_vminnm_o(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_o:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast olt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_o_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_o_rev:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ogt half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vminnm_u(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_u:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ult half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_ule(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_ule:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ule half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_u_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_u_rev:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ugt half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_o(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_o:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ogt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_oge(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_oge:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast oge half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_o_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_o_rev:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast olt half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_ole_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_ole_rev:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ole half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_u(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_u:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ugt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_uge(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_uge:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast uge half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_u_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_u_rev:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp fast ult half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

; known non-NaNs

define half @fp16_vminnm_NNNo(i16 signext %a) {
; CHECK-LABEL: fp16_vminnm_NNNo:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], #1.200000e+01
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast olt half %0, 12.
  %cond1 = select i1 %cmp1, half %0, half 12.
  %cmp2 = fcmp fast olt half 34., %cond1
  %cond2 = select i1 %cmp2, half 34., half %cond1
  ret half %cond2
}

define half @fp16_vminnm_NNNo_rev(i16 signext %a) {
; CHECK-LABEL: fp16_vminnm_NNNo_rev:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast ogt half %0, 56.
  %cond1 = select i1 %cmp1, half 56., half %0
  %cmp2 = fcmp fast ogt half 78., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 78.
  ret half %cond2
}

define half @fp16_vminnm_NNNu(i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_NNNu:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], #1.200000e+01
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast ult half 12., %0
  %cond1 = select i1 %cmp1, half 12., half %0
  %cmp2 = fcmp fast ult half %cond1, 34.
  %cond2 = select i1 %cmp2, half %cond1, half 34.
  ret half %cond2
}

define half @fp16_vminnm_NNNule(i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_NNNule:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast ule half 34., %0
  %cond1 = select i1 %cmp1, half 34., half %0
  %cmp2 = fcmp fast ule half %cond1, 56.
  %cond2 = select i1 %cmp2, half %cond1, half 56.
  ret half %cond2
}

define half @fp16_vminnm_NNNu_rev(i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_NNNu_rev:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vminnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast ugt half 56., %0
  %cond1 = select i1 %cmp1, half %0, half 56.
  %cmp2 = fcmp fast ugt half %cond1, 78.
  %cond2 = select i1 %cmp2, half 78., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNo(i16 signext %a) {
; CHECK-LABEL: fp16_vmaxnm_NNNo:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], #1.200000e+01
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast ogt half %0, 12.
  %cond1 = select i1 %cmp1, half %0, half 12.
  %cmp2 = fcmp fast ogt half 34., %cond1
  %cond2 = select i1 %cmp2, half 34., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNoge(i16 signext %a) {
; CHECK-LABEL: fp16_vmaxnm_NNNoge:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast oge half %0, 34.
  %cond1 = select i1 %cmp1, half %0, half 34.
  %cmp2 = fcmp fast oge half 56., %cond1
  %cond2 = select i1 %cmp2, half 56., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNo_rev(i16 signext %a) {
; CHECK-LABEL: fp16_vmaxnm_NNNo_rev:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast olt half %0, 56.
  %cond1 = select i1 %cmp1, half 56., half %0
  %cmp2 = fcmp fast olt half 78., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 78.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNole_rev(i16 signext %a) {
; CHECK-LABEL: fp16_vmaxnm_NNNole_rev:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast ole half %0, 78.
  %cond1 = select i1 %cmp1, half 78., half %0
  %cmp2 = fcmp fast ole half 90., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 90.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNu(i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_NNNu:
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], #1.200000e+01
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast ugt half 12., %0
  %cond1 = select i1 %cmp1, half 12., half %0
  %cmp2 = fcmp fast ugt half %cond1, 34.
  %cond2 = select i1 %cmp2, half %cond1, half 34.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNuge(i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_NNNuge:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast uge half 34., %0
  %cond1 = select i1 %cmp1, half 34., half %0
  %cmp2 = fcmp fast uge half %cond1, 56.
  %cond2 = select i1 %cmp2, half %cond1, half 56.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNu_rev(i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_NNNu_rev:
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S0:s[0-9]]], r{{.}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
; CHECK:    vldr.16 s2, .LCPI{{.*}}
; CHECK:    vmaxnm.f16 s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp fast ult half 56., %0
  %cond1 = select i1 %cmp1, half %0, half 56.
  %cmp2 = fcmp fast ult half %cond1, 78.
  %cond2 = select i1 %cmp2, half 78., half %cond1
  ret half %cond2
}

define half @fp16_vminmaxnm_0(i16 signext %a) {
; CHECK-LABEL: fp16_vminmaxnm_0:
; CHECK:    vldr.16 s0, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s2, s2, s0
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast olt half %0, 0.
  %cond1 = select i1 %cmp1, half %0, half 0.
  %cmp2 = fcmp fast ogt half %cond1, 0.
  %cond2 = select i1 %cmp2, half %cond1, half 0.
  ret half %cond2
}

define half @fp16_vminmaxnm_neg0(i16 signext %a) {
; CHECK-LABEL: fp16_vminmaxnm_neg0:
; CHECK:    vldr.16 s0, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s2, s2, s0
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast olt half %0, -0.
  %cond1 = select i1 %cmp1, half %0, half -0.
  %cmp2 = fcmp fast ugt half %cond1, -0.
  %cond2 = select i1 %cmp2, half %cond1, half -0.
  ret half %cond2
}

define half @fp16_vminmaxnm_e_0(i16 signext %a) {
; CHECK-LABEL: fp16_vminmaxnm_e_0:
; CHECK:    vldr.16 s0, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s2, s2, s0
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast ule half 0., %0
  %cond1 = select i1 %cmp1, half 0., half %0
  %cmp2 = fcmp fast uge half 0., %cond1
  %cond2 = select i1 %cmp2, half 0., half %cond1
  ret half %cond2
}

define half @fp16_vminmaxnm_e_neg0(i16 signext %a) {
; CHECK-LABEL: fp16_vminmaxnm_e_neg0:
; CHECK:    vldr.16 s0, .LCPI{{.*}}
; CHECK:    vmov.f16 [[S2:s[0-9]]], r{{.}}
; CHECK:    vminnm.f16 s2, s2, s0
; CHECK:    vmaxnm.f16 s0, [[S2]], [[S0]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp fast ule half -0., %0
  %cond1 = select i1 %cmp1, half -0., half %0
  %cmp2 = fcmp fast oge half -0., %cond1
  %cond2 = select i1 %cmp2, half -0., half %cond1
  ret half %cond2
}
