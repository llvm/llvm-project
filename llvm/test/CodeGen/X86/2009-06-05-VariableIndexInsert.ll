; RUN: llc < %s

define <2 x i64> @_mm_insert_epi16(<2 x i64> %a, i32 %b, i32 %imm) nounwind readnone {
entry:
	%conv = bitcast <2 x i64> %a to <8 x i16>		; <<8 x i16>> [#uses=1]
	%conv2 = trunc i32 %b to i16		; <i16> [#uses=1]
	%and = and i32 %imm, 7		; <i32> [#uses=1]
	%vecins = insertelement <8 x i16> %conv, i16 %conv2, i32 %and		; <<8 x i16>> [#uses=1]
	%conv6 = bitcast <8 x i16> %vecins to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %conv6
}
