; RUN: llc < %s -mtriple=armv8-eabi -mattr=+fullfp16 | FileCheck %s
; RUN: llc < %s -mtriple thumbv7a -mattr=+fullfp16 | FileCheck %s

; TODO: we can't pass half-precision arguments as "half" types yet. We do
; that for the time being by passing "float %f.coerce" and the necessary
; bitconverts/truncates. In these tests we pass i16 and use 1 bitconvert, which
; is the shortest way to get a half type. But when we can pass half types, we
; want to use that here.

define half @fp16_vminnm_o(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_o:
; CHECK-NOT: vminnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp olt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_o_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_o_rev:
; CHECK-NOT: vminnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ogt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_u(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_u:
; CHECK-NOT: vminnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ult half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_ule(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_ule:
; CHECK-NOT: vminnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ule half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vminnm_u_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_u_rev:
; CHECK-NOT: vminnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ugt half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_o(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_o:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ogt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_oge(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_oge:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp oge half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_o_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_o_rev:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp olt half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_ole_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_ole_rev:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ole half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

define half @fp16_vmaxnm_u(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_u:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ugt half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_uge(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_uge:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp uge half %0, %1
  %cond = select i1 %cmp, half %0, half %1
  ret half %cond
}

define half @fp16_vmaxnm_u_rev(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: fp16_vmaxnm_u_rev:
; CHECK-NOT: vmaxnm.f16
entry:
  %0 = bitcast i16 %a to half
  %1 = bitcast i16 %b to half
  %cmp = fcmp ult half %0, %1
  %cond = select i1 %cmp, half %1, half %0
  ret half %cond
}

; known non-NaNs

define half @fp16_vminnm_NNNo(i16 signext %a) {
; CHECK-LABEL:  fp16_vminnm_NNNo:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], #1.200000e+01
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vminnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmin.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp olt half %0, 12.
  %cond1 = select i1 %cmp1, half %0, half 12.
  %cmp2 = fcmp olt half 34., %cond1
  %cond2 = select i1 %cmp2, half 34., half %cond1
  ret half %cond2
}

define half @fp16_vminnm_NNNo_rev(i16 signext %a) {
; CHECK-LABEL:  fp16_vminnm_NNNo_rev:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vmin.f16	d0, d1, d0
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vminnm.f16	s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp ogt half %0, 56.
  %cond1 = select i1 %cmp1, half 56., half %0
  %cmp2 = fcmp ogt half 78., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 78.
  ret half %cond2
}

define half @fp16_vminnm_NNNu(i16 signext %b) {
; CHECK-LABEL: fp16_vminnm_NNNu:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], #1.200000e+01
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vminnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmin.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp ult half 12., %0
  %cond1 = select i1 %cmp1, half 12., half %0
  %cmp2 = fcmp ult half %cond1, 34.
  %cond2 = select i1 %cmp2, half %cond1, half 34.
  ret half %cond2
}

define half @fp16_vminnm_NNNule(i16 signext %b) {
; CHECK-LABEL:  fp16_vminnm_NNNule:
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vminnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmin.f16	d0, d1, d0

entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp ule half 34., %0
  %cond1 = select i1 %cmp1, half 34., half %0
  %cmp2 = fcmp ule half %cond1, 56.
  %cond2 = select i1 %cmp2, half %cond1, half 56.
  ret half %cond2
}

define half @fp16_vminnm_NNNu_rev(i16 signext %b) {
; CHECK-LABEL:  fp16_vminnm_NNNu_rev:

; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vmin.f16	d0, d1, d0
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vminnm.f16	s0, [[S0]], [[S2]]

entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp ugt half 56., %0
  %cond1 = select i1 %cmp1, half %0, half 56.
  %cmp2 = fcmp ugt half %cond1, 78.
  %cond2 = select i1 %cmp2, half 78., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNo(i16 signext %a) {
; CHECK-LABEL:  fp16_vmaxnm_NNNo:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], #1.200000e+01
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vmaxnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp ogt half %0, 12.
  %cond1 = select i1 %cmp1, half %0, half 12.
  %cmp2 = fcmp ogt half 34., %cond1
  %cond2 = select i1 %cmp2, half 34., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNoge(i16 signext %a) {
; CHECK-LABEL:  fp16_vmaxnm_NNNoge:
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmaxnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp oge half %0, 34.
  %cond1 = select i1 %cmp1, half %0, half 34.
  %cmp2 = fcmp oge half 56., %cond1
  %cond2 = select i1 %cmp2, half 56., half %cond1
  ret half %cond2
}

define half @fp16_vmaxnm_NNNo_rev(i16 signext %a) {
; CHECK-LABEL:  fp16_vmaxnm_NNNo_rev:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vmax.f16	d0, d1, d0
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmaxnm.f16	s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp olt half %0, 56.
  %cond1 = select i1 %cmp1, half 56., half %0
  %cmp2 = fcmp olt half 78., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 78.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNole_rev(i16 signext %a) {
; CHECK-LABEL:  fp16_vmaxnm_NNNole_rev:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vmax.f16	d0, d1, d0
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmaxnm.f16	s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp ole half %0, 78.
  %cond1 = select i1 %cmp1, half 78., half %0
  %cmp2 = fcmp ole half 90., %cond1
  %cond2 = select i1 %cmp2, half %cond1, half 90.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNu(i16 signext %b) {
; CHECK-LABEL:  fp16_vmaxnm_NNNu:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], #1.200000e+01
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vmaxnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp ugt half 12., %0
  %cond1 = select i1 %cmp1, half 12., half %0
  %cmp2 = fcmp ugt half %cond1, 34.
  %cond2 = select i1 %cmp2, half %cond1, half 34.
  ret half %cond2
}

define half @fp16_vmaxnm_NNNuge(i16 signext %b) {
; CHECK-LABEL:  fp16_vmaxnm_NNNuge:
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S4:s[0-9]]], r{{.}}
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmaxnm.f16	s2, [[S4]], [[S2]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %b to half
  %cmp1 = fcmp uge half 34., %0
  %cond1 = select i1 %cmp1, half 34., half %0
  %cmp2 = fcmp uge half %cond1, 56.
  %cond2 = select i1 %cmp2, half %cond1, half 56.
  ret half %cond2
}

define half @fp16_vminmaxnm_neg0(i16 signext %a) {
; CHECK-LABEL:  fp16_vminmaxnm_neg0:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vminnm.f16	s2, [[S2]], [[S0]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp olt half %0, -0.
  %cond1 = select i1 %cmp1, half %0, half -0.
  %cmp2 = fcmp ugt half %cond1, -0.
  %cond2 = select i1 %cmp2, half %cond1, half -0.
  ret half %cond2
}

define half @fp16_vminmaxnm_e_0(i16 signext %a) {
; CHECK-LABEL:  fp16_vminmaxnm_e_0:
; CHECK:	vldr.16	[[S2:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S0:s[0-9]]], r{{.}}
; CHECK:	vmin.f16	d0, d0, d1
; CHECK:	vmaxnm.f16	s0, [[S0]], [[S2]]
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp nsz ole half 0., %0
  %cond1 = select i1 %cmp1, half 0., half %0
  %cmp2 = fcmp nsz uge half 0., %cond1
  %cond2 = select i1 %cmp2, half 0., half %cond1
  ret half %cond2
}

define half @fp16_vminmaxnm_e_neg0(i16 signext %a) {
; CHECK-LABEL:  fp16_vminmaxnm_e_neg0:
; CHECK:	vldr.16	[[S0:s[0-9]]], .LCPI{{.*}}
; CHECK:	vmov.f16	[[S2:s[0-9]]], r{{.}}
; CHECK:	vminnm.f16	s2, [[S2]], [[S0]]
; CHECK:	vmax.f16	d0, d1, d0
entry:
  %0 = bitcast i16 %a to half
  %cmp1 = fcmp nsz ule half -0., %0
  %cond1 = select i1 %cmp1, half -0., half %0
  %cmp2 = fcmp nsz oge half -0., %cond1
  %cond2 = select i1 %cmp2, half -0., half %cond1
  ret half %cond2
}
