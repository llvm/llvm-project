; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep CPI

define <2 x i64> @t1(<2 x i64> %b1, <2 x i64> %c) nounwind  {
	%tmp1 = bitcast <2 x i64> %b1 to <8 x i16>
	%tmp2 = tail call <8 x i16> @llvm.x86.sse2.psrl.w( <8 x i16> %tmp1, <8 x i16> bitcast (<4 x i32> < i32 14, i32 undef, i32 undef, i32 undef > to <8 x i16>) ) nounwind readnone
	%tmp3 = bitcast <8 x i16> %tmp2 to <2 x i64>
	ret <2 x i64> %tmp3
}

define <4 x i32> @t2(<2 x i64> %b1, <2 x i64> %c) nounwind  {
	%tmp1 = bitcast <2 x i64> %b1 to <4 x i32>
	%tmp2 = tail call <4 x i32> @llvm.x86.sse2.psll.d( <4 x i32> %tmp1, <4 x i32> < i32 14, i32 undef, i32 undef, i32 undef > ) nounwind readnone
	ret <4 x i32> %tmp2
}

declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>) nounwind readnone 
declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>) nounwind readnone 
