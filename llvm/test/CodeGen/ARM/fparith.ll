; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 | FileCheck %s

define float @f1(float %a, float %b) {
;CHECK-LABEL: f1:
;CHECK: vadd.f32
entry:
	%tmp = fadd float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f2(double %a, double %b) {
;CHECK-LABEL: f2:
;CHECK: vadd.f64
entry:
	%tmp = fadd double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f3(float %a, float %b) {
;CHECK-LABEL: f3:
;CHECK: vmul.f32
entry:
	%tmp = fmul float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f4(double %a, double %b) {
;CHECK-LABEL: f4:
;CHECK: vmul.f64
entry:
	%tmp = fmul double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f5(float %a, float %b) {
;CHECK-LABEL: f5:
;CHECK: vsub.f32
entry:
	%tmp = fsub float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f6(double %a, double %b) {
;CHECK-LABEL: f6:
;CHECK: vsub.f64
entry:
	%tmp = fsub double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f7(float %a) {
;CHECK-LABEL: f7:
;CHECK: eor
entry:
	%tmp1 = fsub float -0.000000e+00, %a		; <float> [#uses=1]
	ret float %tmp1
}

define arm_aapcs_vfpcc double @f8(double %a) {
;CHECK-LABEL: f8:
;CHECK: vneg.f64
entry:
	%tmp1 = fsub double -0.000000e+00, %a		; <double> [#uses=1]
	ret double %tmp1
}

define float @f9(float %a, float %b) {
;CHECK-LABEL: f9:
;CHECK: vdiv.f32
entry:
	%tmp1 = fdiv float %a, %b		; <float> [#uses=1]
	ret float %tmp1
}

define double @f10(double %a, double %b) {
;CHECK-LABEL: f10:
;CHECK: vdiv.f64
entry:
	%tmp1 = fdiv double %a, %b		; <double> [#uses=1]
	ret double %tmp1
}

define float @f11(float %a) {
;CHECK-LABEL: f11:
;CHECK: bic
entry:
	%tmp1 = call float @fabsf( float %a ) readnone	; <float> [#uses=1]
	ret float %tmp1
}

declare float @fabsf(float)

define arm_aapcs_vfpcc double @f12(double %a) {
;CHECK-LABEL: f12:
;CHECK: vabs.f64
entry:
	%tmp1 = call double @fabs( double %a ) readnone	; <double> [#uses=1]
	ret double %tmp1
}

declare double @fabs(double)
