; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movq
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep mov | count 3

define <2 x i64> @t1(<2 x i64>* %ptr) nounwind  {
	%tmp45 = bitcast <2 x i64>* %ptr to <2 x i32>*
	%tmp615 = load <2 x i32>* %tmp45
	%tmp7 = bitcast <2 x i32> %tmp615 to i64
	%tmp8 = insertelement <2 x i64> zeroinitializer, i64 %tmp7, i32 0
	ret <2 x i64> %tmp8
}

define <2 x i64> @t2(i64 %x) nounwind  {
	%tmp717 = bitcast i64 %x to double
	%tmp8 = insertelement <2 x double> undef, double %tmp717, i32 0
	%tmp9 = insertelement <2 x double> %tmp8, double 0.000000e+00, i32 1
	%tmp11 = bitcast <2 x double> %tmp9 to <2 x i64>
	ret <2 x i64> %tmp11
}
