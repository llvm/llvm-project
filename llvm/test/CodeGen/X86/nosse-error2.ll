; RUN: not llc < %s -mcpu=i686 -mattr=-sse 2>&1 | FileCheck --check-prefix NOSSE %s
; RUN: llc < %s -mcpu=i686 -mattr=+sse | FileCheck %s

; NOSSE: {{SSE register return with SSE disabled}}

; CHECK: xmm

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-unknown-linux-gnu"
@f = external global float		; <float*> [#uses=4]
@d = external global double		; <double*> [#uses=4]

define void @test() nounwind {
entry:
	%0 = load float, float* @f, align 4		; <float> [#uses=1]
	%1 = tail call inreg float @foo1(float inreg %0) nounwind		; <float> [#uses=1]
	store float %1, float* @f, align 4
	%2 = load double, double* @d, align 8		; <double> [#uses=1]
	%3 = tail call inreg double @foo2(double inreg %2) nounwind		; <double> [#uses=1]
	store double %3, double* @d, align 8
	%4 = load float, float* @f, align 4		; <float> [#uses=1]
	%5 = tail call inreg float @foo3(float inreg %4) nounwind		; <float> [#uses=1]
	store float %5, float* @f, align 4
	%6 = load double, double* @d, align 8		; <double> [#uses=1]
	%7 = tail call inreg double @foo4(double inreg %6) nounwind		; <double> [#uses=1]
	store double %7, double* @d, align 8
	ret void
}

declare inreg float @foo1(float inreg)

declare inreg double @foo2(double inreg)

declare inreg float @foo3(float inreg)

declare inreg double @foo4(double inreg)
