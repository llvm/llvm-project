; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep test

define float @tst1(float %a, float %b) {
	%tmp = tail call float @copysignf( float %b, float %a )
	ret float %tmp
}

define double @tst2(double %a, float %b, float %c) {
	%tmp1 = fadd float %b, %c
	%tmp2 = fpext float %tmp1 to double
	%tmp = tail call double @copysign( double %a, double %tmp2 )
	ret double %tmp
}

declare float @copysignf(float, float)
declare double @copysign(double, double)
