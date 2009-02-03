; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep movz

define <2 x i64> @doload64(i16 signext  %x) nounwind  {
entry:
	%tmp36 = insertelement <8 x i16> undef, i16 %x, i32 0		; <<8 x i16>> [#uses=1]
	%tmp37 = insertelement <8 x i16> %tmp36, i16 %x, i32 1		; <<8 x i16>> [#uses=1]
	%tmp38 = insertelement <8 x i16> %tmp37, i16 %x, i32 2		; <<8 x i16>> [#uses=1]
	%tmp39 = insertelement <8 x i16> %tmp38, i16 %x, i32 3		; <<8 x i16>> [#uses=1]
	%tmp40 = insertelement <8 x i16> %tmp39, i16 %x, i32 4		; <<8 x i16>> [#uses=1]
	%tmp41 = insertelement <8 x i16> %tmp40, i16 %x, i32 5		; <<8 x i16>> [#uses=1]
	%tmp42 = insertelement <8 x i16> %tmp41, i16 %x, i32 6		; <<8 x i16>> [#uses=1]
	%tmp43 = insertelement <8 x i16> %tmp42, i16 %x, i32 7		; <<8 x i16>> [#uses=1]
	%tmp46 = bitcast <8 x i16> %tmp43 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp46
}
