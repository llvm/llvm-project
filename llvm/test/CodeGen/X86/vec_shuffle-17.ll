; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s
; CHECK-NOT: xor
; CHECK: movd {{%rdi|%rcx}}, %xmm0
; CHECK-NOT: xor
; PR2108

define <2 x i64> @doload64(i64 %x) nounwind  {
entry:
	%tmp717 = bitcast i64 %x to double		; <double> [#uses=1]
	%tmp8 = insertelement <2 x double> undef, double %tmp717, i32 0		; <<2 x double>> [#uses=1]
	%tmp9 = insertelement <2 x double> %tmp8, double 0.000000e+00, i32 1		; <<2 x double>> [#uses=1]
	%tmp11 = bitcast <2 x double> %tmp9 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp11
}

