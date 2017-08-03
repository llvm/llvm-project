; REQUIRES: asserts
; RUN: llc < %s -march=x86 -mcpu=yonah -stats 2>&1 | \
; RUN:   not grep "Number of register spills"
; END.


define i32 @foo(<4 x float>* %a, <4 x float>* %b, <4 x float>* %c, <4 x float>* %d) {
	%tmp44 = load <4 x float>, <4 x float>* %a		; <<4 x float>> [#uses=9]
	%tmp46 = load <4 x float>, <4 x float>* %b		; <<4 x float>> [#uses=1]
	%tmp48 = load <4 x float>, <4 x float>* %c		; <<4 x float>> [#uses=1]
	%tmp50 = load <4 x float>, <4 x float>* %d		; <<4 x float>> [#uses=1]
	%tmp51 = bitcast <4 x float> %tmp44 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp = shufflevector <4 x i32> %tmp51, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>> [#uses=2]
	%tmp52 = bitcast <4 x i32> %tmp to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp60 = xor <4 x i32> %tmp, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	%tmp61 = bitcast <4 x i32> %tmp60 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp74 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp52, <4 x float> %tmp44, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp75 = bitcast <4 x float> %tmp74 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp88 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp44, <4 x float> %tmp61, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp89 = bitcast <4 x float> %tmp88 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp98 = tail call <8 x i16> @llvm.x86.sse2.packssdw.128( <4 x i32> %tmp75, <4 x i32> %tmp89 )		; <<4 x i32>> [#uses=1]
	%tmp102 = bitcast <8 x i16> %tmp98 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.1 = shufflevector <8 x i16> %tmp102, <8 x i16> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 6, i32 5, i32 4, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp105 = shufflevector <8 x i16> %tmp.upgrd.1, <8 x i16> undef, <8 x i32> < i32 2, i32 1, i32 0, i32 3, i32 4, i32 5, i32 6, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp105.upgrd.2 = bitcast <8 x i16> %tmp105 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp105.upgrd.2, <4 x float>* %a
	%tmp108 = bitcast <4 x float> %tmp46 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp109 = shufflevector <4 x i32> %tmp108, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>> [#uses=2]
	%tmp109.upgrd.3 = bitcast <4 x i32> %tmp109 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp119 = xor <4 x i32> %tmp109, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	%tmp120 = bitcast <4 x i32> %tmp119 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp133 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp109.upgrd.3, <4 x float> %tmp44, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp134 = bitcast <4 x float> %tmp133 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp147 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp44, <4 x float> %tmp120, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp148 = bitcast <4 x float> %tmp147 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp159 = tail call <8 x i16> @llvm.x86.sse2.packssdw.128( <4 x i32> %tmp134, <4 x i32> %tmp148 )		; <<4 x i32>> [#uses=1]
	%tmp163 = bitcast <8 x i16> %tmp159 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp164 = shufflevector <8 x i16> %tmp163, <8 x i16> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 6, i32 5, i32 4, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp166 = shufflevector <8 x i16> %tmp164, <8 x i16> undef, <8 x i32> < i32 2, i32 1, i32 0, i32 3, i32 4, i32 5, i32 6, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp166.upgrd.4 = bitcast <8 x i16> %tmp166 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp166.upgrd.4, <4 x float>* %b
	%tmp169 = bitcast <4 x float> %tmp48 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp170 = shufflevector <4 x i32> %tmp169, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>> [#uses=2]
	%tmp170.upgrd.5 = bitcast <4 x i32> %tmp170 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp180 = xor <4 x i32> %tmp170, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	%tmp181 = bitcast <4 x i32> %tmp180 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp194 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp170.upgrd.5, <4 x float> %tmp44, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp195 = bitcast <4 x float> %tmp194 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp208 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp44, <4 x float> %tmp181, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp209 = bitcast <4 x float> %tmp208 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp220 = tail call <8 x i16> @llvm.x86.sse2.packssdw.128( <4 x i32> %tmp195, <4 x i32> %tmp209 )		; <<4 x i32>> [#uses=1]
	%tmp224 = bitcast <8 x i16> %tmp220 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp225 = shufflevector <8 x i16> %tmp224, <8 x i16> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 6, i32 5, i32 4, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp227 = shufflevector <8 x i16> %tmp225, <8 x i16> undef, <8 x i32> < i32 2, i32 1, i32 0, i32 3, i32 4, i32 5, i32 6, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp227.upgrd.6 = bitcast <8 x i16> %tmp227 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp227.upgrd.6, <4 x float>* %c
	%tmp230 = bitcast <4 x float> %tmp50 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp231 = shufflevector <4 x i32> %tmp230, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>> [#uses=2]
	%tmp231.upgrd.7 = bitcast <4 x i32> %tmp231 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp241 = xor <4 x i32> %tmp231, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	%tmp242 = bitcast <4 x i32> %tmp241 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp255 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp231.upgrd.7, <4 x float> %tmp44, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp256 = bitcast <4 x float> %tmp255 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp269 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp44, <4 x float> %tmp242, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp270 = bitcast <4 x float> %tmp269 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp281 = tail call <8 x i16> @llvm.x86.sse2.packssdw.128( <4 x i32> %tmp256, <4 x i32> %tmp270 )		; <<4 x i32>> [#uses=1]
	%tmp285 = bitcast <8 x i16> %tmp281 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp286 = shufflevector <8 x i16> %tmp285, <8 x i16> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 6, i32 5, i32 4, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp288 = shufflevector <8 x i16> %tmp286, <8 x i16> undef, <8 x i32> < i32 2, i32 1, i32 0, i32 3, i32 4, i32 5, i32 6, i32 7 >		; <<8 x i16>> [#uses=1]
	%tmp288.upgrd.8 = bitcast <8 x i16> %tmp288 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp288.upgrd.8, <4 x float>* %d
	ret i32 0
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8)

declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>)
