; RUN: llc < %s -mtriple=i686-- -mattr=+sse,+sse2 | FileCheck %s

define float @min1(float %x, float %y) {
; CHECK-LABEL: min1
; CHECK: mins
	%tmp = fcmp olt float %x, %y
	%retval = select i1 %tmp, float %x, float %y
	ret float %retval
}

define double @min2(double %x, double %y) {
; CHECK-LABEL: min2
; CHECK: mins
	%tmp = fcmp olt double %x, %y
	%retval = select i1 %tmp, double %x, double %y
	ret double %retval
}

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)
define <4 x float> @min3(float %x, float %y) {
; CHECK-LABEL: min3
; CHECK: mins
	%vec0 = insertelement <4 x float> undef, float %x, i32 0
	%vec1 = insertelement <4 x float> undef, float %y, i32 0
	%retval = tail call <4 x float> @llvm.x86.sse.min.ss(<4 x float> %vec0, <4 x float> %vec1)
	ret <4 x float> %retval
}

define float @max1(float %x, float %y) {
; CHECK-LABEL: max1
; CHECK: maxs
	%tmp = fcmp uge float %x, %y
	%retval = select i1 %tmp, float %x, float %y
	ret float %retval
}

define double @max2(double %x, double %y) {
; CHECK-LABEL: max2
; CHECK: maxs
	%tmp = fcmp uge double %x, %y
	%retval = select i1 %tmp, double %x, double %y
	ret double %retval
}

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>)
define <4 x float> @max3(float %x, float %y) {
; CHECK-LABEL: max3
; CHECK: maxs
	%vec0 = insertelement <4 x float> undef, float %x, i32 0
	%vec1 = insertelement <4 x float> undef, float %y, i32 0
	%retval = tail call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %vec0, <4 x float> %vec1)
	ret <4 x float> %retval
}
