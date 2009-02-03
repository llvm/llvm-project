; RUN: llvm-as < %s | llc -march=x86-64 | grep {movd.*%rdi, %xmm0}
; RUN: llvm-as < %s | llc -march=x86-64 | not grep xor
; PR2108

define <2 x i64> @doload64(i64 %x) nounwind  {
entry:
	%tmp717 = bitcast i64 %x to double		; <double> [#uses=1]
	%tmp8 = insertelement <2 x double> undef, double %tmp717, i32 0		; <<2 x double>> [#uses=1]
	%tmp9 = insertelement <2 x double> %tmp8, double 0.000000e+00, i32 1		; <<2 x double>> [#uses=1]
	%tmp11 = bitcast <2 x double> %tmp9 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp11
}

