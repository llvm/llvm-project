; This should fold the "vcmpbfp." and "vcmpbfp" instructions into a single
; "vcmpbfp.".
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vcmpbfp | count 1


define void @test(<4 x float>* %x, <4 x float>* %y, i32* %P) {
entry:
	%tmp = load <4 x float>* %x		; <<4 x float>> [#uses=1]
	%tmp2 = load <4 x float>* %y		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = call i32 @llvm.ppc.altivec.vcmpbfp.p( i32 1, <4 x float> %tmp, <4 x float> %tmp2 )		; <i32> [#uses=1]
	%tmp4 = load <4 x float>* %x		; <<4 x float>> [#uses=1]
	%tmp6 = load <4 x float>* %y		; <<4 x float>> [#uses=1]
	%tmp.upgrd.2 = call <4 x i32> @llvm.ppc.altivec.vcmpbfp( <4 x float> %tmp4, <4 x float> %tmp6 )		; <<4 x i32>> [#uses=1]
	%tmp7 = bitcast <4 x i32> %tmp.upgrd.2 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp7, <4 x float>* %x
	store i32 %tmp.upgrd.1, i32* %P
	ret void
}

declare i32 @llvm.ppc.altivec.vcmpbfp.p(i32, <4 x float>, <4 x float>)

declare <4 x i32> @llvm.ppc.altivec.vcmpbfp(<4 x float>, <4 x float>)
