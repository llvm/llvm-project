; RUN: llc < %s -march=x86 -mattr=+sse,+sse2 | \
; RUN:   grep mins | count 3
; RUN: llc < %s -march=x86 -mattr=+sse,+sse2 | \
; RUN:   grep maxs | count 2

declare i1 @llvm.isunordered.f64(double, double)

declare i1 @llvm.isunordered.f32(float, float)

define float @min1(float %x, float %y) {
	%tmp = fcmp olt float %x, %y		; <i1> [#uses=1]
	%retval = select i1 %tmp, float %x, float %y		; <float> [#uses=1]
	ret float %retval
}

define double @min2(double %x, double %y) {
	%tmp = fcmp olt double %x, %y		; <i1> [#uses=1]
	%retval = select i1 %tmp, double %x, double %y		; <double> [#uses=1]
	ret double %retval
}

define float @max1(float %x, float %y) {
	%tmp = fcmp oge float %x, %y		; <i1> [#uses=1]
	%tmp2 = fcmp uno float %x, %y		; <i1> [#uses=1]
	%tmp3 = or i1 %tmp2, %tmp		; <i1> [#uses=1]
	%retval = select i1 %tmp3, float %x, float %y		; <float> [#uses=1]
	ret float %retval
}

define double @max2(double %x, double %y) {
	%tmp = fcmp oge double %x, %y		; <i1> [#uses=1]
	%tmp2 = fcmp uno double %x, %y		; <i1> [#uses=1]
	%tmp3 = or i1 %tmp2, %tmp		; <i1> [#uses=1]
	%retval = select i1 %tmp3, double %x, double %y		; <double> [#uses=1]
	ret double %retval
}

define <4 x float> @min3(float %tmp37) {
	%tmp375 = insertelement <4 x float> undef, float %tmp37, i32 0		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp375, <4 x float> < float 6.553500e+04, float undef, float undef, float undef > )		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp48
}

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)
