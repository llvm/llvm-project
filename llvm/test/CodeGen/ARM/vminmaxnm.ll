; RUN: llc < %s -mtriple armv8 -mattr=+neon,+fp-armv8 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s

; scalars

define float @fp-armv8_vminnm_o(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_o":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
  %cmp = fcmp fast olt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define double @fp-armv8_vminnm_ole(double %a, double %b) {
; CHECK-LABEL: "fp-armv8_vminnm_ole":
; CHECK-NOT: vcmp
; CHECK: vminnm.f64
  %cmp = fcmp fast ole double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond
}

define float @fp-armv8_vminnm_o_rev(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_o_rev":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
  %cmp = fcmp fast ogt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vminnm_oge_rev(double %a, double %b) {
; CHECK-LABEL: "fp-armv8_vminnm_oge_rev":
; CHECK-NOT: vcmp
; CHECK: vminnm.f64
  %cmp = fcmp fast oge double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}

define float @fp-armv8_vminnm_u(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_u":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
  %cmp = fcmp fast ult float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vminnm_ule(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_ule":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
  %cmp = fcmp fast ule float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vminnm_u_rev(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_u_rev":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
  %cmp = fcmp fast ugt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vminnm_uge_rev(double %a, double %b) {
; CHECK-LABEL: "fp-armv8_vminnm_uge_rev":
; CHECK-NOT: vcmp
; CHECK: vminnm.f64
  %cmp = fcmp fast uge double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}

define float @fp-armv8_vmaxnm_o(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_o":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast ogt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_oge(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_oge":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast oge float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_o_rev(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_o_rev":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast olt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define float @fp-armv8_vmaxnm_ole_rev(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_ole_rev":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast ole float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define float @fp-armv8_vmaxnm_u(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_u":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast ugt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_uge(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_uge":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast uge float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_u_rev(float %a, float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_u_rev":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f32
  %cmp = fcmp fast ult float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vmaxnm_ule_rev(double %a, double %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_ule_rev":
; CHECK-NOT: vcmp
; CHECK: vmaxnm.f64
  %cmp = fcmp fast ule double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}

; known non-NaNs

define float @fp-armv8_vminnm_NNNo(float %a) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNo":
; CHECK: vminnm.f32
; CHECK: vminnm.f32
  %cmp1 = fcmp fast olt float %a, 12.
  %cond1 = select i1 %cmp1, float %a, float 12.
  %cmp2 = fcmp fast olt float 34., %cond1
  %cond2 = select i1 %cmp2, float 34., float %cond1
  ret float %cond2
}

define double @fp-armv8_vminnm_NNNole(double %a) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNole":
; CHECK: vminnm.f64
; CHECK: vminnm.f64
  %cmp1 = fcmp fast ole double %a, 34.
  %cond1 = select i1 %cmp1, double %a, double 34.
  %cmp2 = fcmp fast ole double 56., %cond1
  %cond2 = select i1 %cmp2, double 56., double %cond1
  ret double %cond2
}

define float @fp-armv8_vminnm_NNNo_rev(float %a) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNo_rev":
; CHECK: vminnm.f32
; CHECK: vminnm.f32
  %cmp1 = fcmp fast ogt float %a, 56.
  %cond1 = select i1 %cmp1, float 56., float %a
  %cmp2 = fcmp fast ogt float 78., %cond1
  %cond2 = select i1 %cmp2, float %cond1, float 78.
  ret float %cond2
}

define double @fp-armv8_vminnm_NNNoge_rev(double %a) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNoge_rev":
; CHECK: vminnm.f64
; CHECK: vminnm.f64
  %cmp1 = fcmp fast oge double %a, 78.
  %cond1 = select i1 %cmp1, double 78., double %a
  %cmp2 = fcmp fast oge double 90., %cond1
  %cond2 = select i1 %cmp2, double %cond1, double 90.
  ret double %cond2
}

define float @fp-armv8_vminnm_NNNu(float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNu":
; CHECK: vminnm.f32
; CHECK: vminnm.f32
  %cmp1 = fcmp fast ult float 12., %b
  %cond1 = select i1 %cmp1, float 12., float %b
  %cmp2 = fcmp fast ult float %cond1, 34.
  %cond2 = select i1 %cmp2, float %cond1, float 34.
  ret float %cond2
}

define float @fp-armv8_vminnm_NNNule(float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNule":
; CHECK: vminnm.f32
; CHECK: vminnm.f32
  %cmp1 = fcmp fast ule float 34., %b
  %cond1 = select i1 %cmp1, float 34., float %b
  %cmp2 = fcmp fast ule float %cond1, 56.
  %cond2 = select i1 %cmp2, float %cond1, float 56.
  ret float %cond2
}

define float @fp-armv8_vminnm_NNNu_rev(float %b) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNu_rev":
; CHECK: vminnm.f32
; CHECK: vminnm.f32
  %cmp1 = fcmp fast ugt float 56., %b
  %cond1 = select i1 %cmp1, float %b, float 56.
  %cmp2 = fcmp fast ugt float %cond1, 78.
  %cond2 = select i1 %cmp2, float 78., float %cond1
  ret float %cond2
}

define double @fp-armv8_vminnm_NNNuge_rev(double %b) {
; CHECK-LABEL: "fp-armv8_vminnm_NNNuge_rev":
; CHECK: vminnm.f64
; CHECK: vminnm.f64
  %cmp1 = fcmp fast uge double 78., %b
  %cond1 = select i1 %cmp1, double %b, double 78.
  %cmp2 = fcmp fast uge double %cond1, 90.
  %cond2 = select i1 %cmp2, double 90., double %cond1
  ret double %cond2
}

define float @fp-armv8_vmaxnm_NNNo(float %a) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNo":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ogt float %a, 12.
  %cond1 = select i1 %cmp1, float %a, float 12.
  %cmp2 = fcmp fast ogt float 34., %cond1
  %cond2 = select i1 %cmp2, float 34., float %cond1
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNoge(float %a) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNoge":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast oge float %a, 34.
  %cond1 = select i1 %cmp1, float %a, float 34.
  %cmp2 = fcmp fast oge float 56., %cond1
  %cond2 = select i1 %cmp2, float 56., float %cond1
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNo_rev(float %a) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNo_rev":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast olt float %a, 56.
  %cond1 = select i1 %cmp1, float 56., float %a
  %cmp2 = fcmp fast olt float 78., %cond1
  %cond2 = select i1 %cmp2, float %cond1, float 78.
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNole_rev(float %a) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNole_rev":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ole float %a, 78.
  %cond1 = select i1 %cmp1, float 78., float %a
  %cmp2 = fcmp fast ole float 90., %cond1
  %cond2 = select i1 %cmp2, float %cond1, float 90.
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNu(float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNu":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ugt float 12., %b
  %cond1 = select i1 %cmp1, float 12., float %b
  %cmp2 = fcmp fast ugt float %cond1, 34.
  %cond2 = select i1 %cmp2, float %cond1, float 34.
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNuge(float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNuge":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast uge float 34., %b
  %cond1 = select i1 %cmp1, float 34., float %b
  %cmp2 = fcmp fast uge float %cond1, 56.
  %cond2 = select i1 %cmp2, float %cond1, float 56.
  ret float %cond2
}

define float @fp-armv8_vmaxnm_NNNu_rev(float %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNu_rev":
; CHECK: vmaxnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ult float 56., %b
  %cond1 = select i1 %cmp1, float %b, float 56.
  %cmp2 = fcmp fast ult float %cond1, 78.
  %cond2 = select i1 %cmp2, float 78., float %cond1
  ret float %cond2
}

define double @fp-armv8_vmaxnm_NNNule_rev( double %b) {
; CHECK-LABEL: "fp-armv8_vmaxnm_NNNule_rev":
; CHECK: vmaxnm.f64
; CHECK: vmaxnm.f64
  %cmp1 = fcmp fast ule double 78., %b
  %cond1 = select i1 %cmp1, double %b, double 78.
  %cmp2 = fcmp fast ule double %cond1, 90.
  %cond2 = select i1 %cmp2, double 90., double %cond1
  ret double %cond2
}

define float @fp-armv8_vminmaxnm_0(float %a) {
; CHECK-LABEL: "fp-armv8_vminmaxnm_0":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast olt float %a, 0.
  %cond1 = select i1 %cmp1, float %a, float 0.
  %cmp2 = fcmp fast ogt float %cond1, 0.
  %cond2 = select i1 %cmp2, float %cond1, float 0.
  ret float %cond2
}

define float @fp-armv8_vminmaxnm_neg0(float %a) {
; CHECK-LABEL: "fp-armv8_vminmaxnm_neg0":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast olt float %a, -0.
  %cond1 = select i1 %cmp1, float %a, float -0.
  %cmp2 = fcmp fast ugt float %cond1, -0.
  %cond2 = select i1 %cmp2, float %cond1, float -0.
  ret float %cond2
}

define float @fp-armv8_vminmaxnm_e_0(float %a) {
; CHECK-LABEL: "fp-armv8_vminmaxnm_e_0":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ule float 0., %a
  %cond1 = select i1 %cmp1, float 0., float %a
  %cmp2 = fcmp fast uge float 0., %cond1
  %cond2 = select i1 %cmp2, float 0., float %cond1
  ret float %cond2
}

define float @fp-armv8_vminmaxnm_e_neg0(float %a) {
; CHECK-LABEL: "fp-armv8_vminmaxnm_e_neg0":
; CHECK-NOT: vcmp
; CHECK: vminnm.f32
; CHECK: vmaxnm.f32
  %cmp1 = fcmp fast ule float -0., %a
  %cond1 = select i1 %cmp1, float -0., float %a
  %cmp2 = fcmp fast oge float -0., %cond1
  %cond2 = select i1 %cmp2, float -0., float %cond1
  ret float %cond2
}

declare <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float>, <2 x float>) nounwind readnone
