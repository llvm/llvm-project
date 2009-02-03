; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep mullw
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vmsumuhm

define <4 x i32> @test_v4i32(<4 x i32>* %X, <4 x i32>* %Y) {
	%tmp = load <4 x i32>* %X		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x i32>* %Y		; <<4 x i32>> [#uses=1]
	%tmp3 = mul <4 x i32> %tmp, %tmp2		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp3
}

define <8 x i16> @test_v8i16(<8 x i16>* %X, <8 x i16>* %Y) {
	%tmp = load <8 x i16>* %X		; <<8 x i16>> [#uses=1]
	%tmp2 = load <8 x i16>* %Y		; <<8 x i16>> [#uses=1]
	%tmp3 = mul <8 x i16> %tmp, %tmp2		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %tmp3
}

define <16 x i8> @test_v16i8(<16 x i8>* %X, <16 x i8>* %Y) {
	%tmp = load <16 x i8>* %X		; <<16 x i8>> [#uses=1]
	%tmp2 = load <16 x i8>* %Y		; <<16 x i8>> [#uses=1]
	%tmp3 = mul <16 x i8> %tmp, %tmp2		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %tmp3
}
