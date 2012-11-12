; RUN: llc < %s -march=x86 -mattr=+sse2 | grep psllq
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep psraw
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movd | count 2

define <2 x i64> @t1(<2 x i64> %x1, i32 %bits) nounwind  {
entry:
	%tmp3 = tail call <2 x i64> @llvm.x86.sse2.pslli.q( <2 x i64> %x1, i32 %bits ) nounwind readnone 		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp3
}

define <2 x i64> @t2(<2 x i64> %x1) nounwind  {
entry:
	%tmp3 = tail call <2 x i64> @llvm.x86.sse2.pslli.q( <2 x i64> %x1, i32 10 ) nounwind readnone 		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp3
}

define <2 x i64> @t3(<2 x i64> %x1, i32 %bits) nounwind  {
entry:
	%tmp2 = bitcast <2 x i64> %x1 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp4 = tail call <8 x i16> @llvm.x86.sse2.psrai.w( <8 x i16> %tmp2, i32 %bits ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp5 = bitcast <8 x i16> %tmp4 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp5
}

declare <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16>, i32) nounwind readnone 
declare <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64>, i32) nounwind readnone 
