; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 > %t
; RUN: grep fadds %t
; RUN: grep faddd %t
; RUN: grep fmuls %t
; RUN: grep fmuld %t
; RUN: grep eor %t
; RUN: grep fnegd %t
; RUN: grep fdivs %t
; RUN: grep fdivd %t

define float @f1(float %a, float %b) {
entry:
	%tmp = add float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f2(double %a, double %b) {
entry:
	%tmp = add double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f3(float %a, float %b) {
entry:
	%tmp = mul float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f4(double %a, double %b) {
entry:
	%tmp = mul double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f5(float %a, float %b) {
entry:
	%tmp = sub float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f6(double %a, double %b) {
entry:
	%tmp = sub double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f7(float %a) {
entry:
	%tmp1 = sub float -0.000000e+00, %a		; <float> [#uses=1]
	ret float %tmp1
}

define double @f8(double %a) {
entry:
	%tmp1 = sub double -0.000000e+00, %a		; <double> [#uses=1]
	ret double %tmp1
}

define float @f9(float %a, float %b) {
entry:
	%tmp1 = fdiv float %a, %b		; <float> [#uses=1]
	ret float %tmp1
}

define double @f10(double %a, double %b) {
entry:
	%tmp1 = fdiv double %a, %b		; <double> [#uses=1]
	ret double %tmp1
}

define float @f11(float %a) {
entry:
	%tmp1 = call float @fabsf( float %a )		; <float> [#uses=1]
	ret float %tmp1
}

declare float @fabsf(float)

define double @f12(double %a) {
entry:
	%tmp1 = call double @fabs( double %a )		; <double> [#uses=1]
	ret double %tmp1
}

declare double @fabs(double)
