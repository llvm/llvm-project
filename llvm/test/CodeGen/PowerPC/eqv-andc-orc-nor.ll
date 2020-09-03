; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep eqv | count 3
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | \
; RUN:   grep andc | count 3
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep orc | count 2
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | \
; RUN:   grep nor | count 2
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep nand | count 1

define i32 @EQV1(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %X, %Y		; <i32> [#uses=1]
	%B = xor i32 %A, -1		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @EQV2(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %X, -1		; <i32> [#uses=1]
	%B = xor i32 %A, %Y		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @EQV3(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %X, -1		; <i32> [#uses=1]
	%B = xor i32 %Y, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @ANDC1(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %Y, -1		; <i32> [#uses=1]
	%B = and i32 %X, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @ANDC2(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %X, -1		; <i32> [#uses=1]
	%B = and i32 %A, %Y		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @ORC1(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %Y, -1		; <i32> [#uses=1]
	%B = or i32 %X, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @ORC2(i32 %X, i32 %Y) nounwind {
	%A = xor i32 %X, -1		; <i32> [#uses=1]
	%B = or i32 %A, %Y		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @NOR1(i32 %X) nounwind {
	%Y = xor i32 %X, -1		; <i32> [#uses=1]
	ret i32 %Y
}

define i32 @NOR2(i32 %X, i32 %Y) nounwind {
	%Z = or i32 %X, %Y		; <i32> [#uses=1]
	%R = xor i32 %Z, -1		; <i32> [#uses=1]
	ret i32 %R
}

define i32 @NAND1(i32 %X, i32 %Y) nounwind {
	%Z = and i32 %X, %Y		; <i32> [#uses=1]
	%W = xor i32 %Z, -1		; <i32> [#uses=1]
	ret i32 %W
}

define void @VNOR(<4 x float>* %P, <4 x float>* %Q) nounwind {
	%tmp = load <4 x float>, <4 x float>* %P		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = bitcast <4 x float> %tmp to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x float>, <4 x float>* %Q		; <<4 x float>> [#uses=1]
	%tmp2.upgrd.2 = bitcast <4 x float> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp3 = or <4 x i32> %tmp.upgrd.1, %tmp2.upgrd.2		; <<4 x i32>> [#uses=1]
	%tmp4 = xor <4 x i32> %tmp3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp4.upgrd.3 = bitcast <4 x i32> %tmp4 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp4.upgrd.3, <4 x float>* %P
	ret void
}

define void @VANDC(<4 x float>* %P, <4 x float>* %Q) nounwind {
	%tmp = load <4 x float>, <4 x float>* %P		; <<4 x float>> [#uses=1]
	%tmp.upgrd.4 = bitcast <4 x float> %tmp to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x float>, <4 x float>* %Q		; <<4 x float>> [#uses=1]
	%tmp2.upgrd.5 = bitcast <4 x float> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp4 = xor <4 x i32> %tmp2.upgrd.5, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp3 = and <4 x i32> %tmp.upgrd.4, %tmp4		; <<4 x i32>> [#uses=1]
	%tmp4.upgrd.6 = bitcast <4 x i32> %tmp3 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp4.upgrd.6, <4 x float>* %P
	ret void
}
