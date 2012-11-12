; RUN: llc < %s -march=x86 -mattr=+sse41 > %t
; RUN: grep pinsrd %t | count 1

define <4 x i32> @var_insert2(<4 x i32> %x, i32 %val, i32 %idx) nounwind  {
entry:
	%tmp3 = insertelement <4 x i32> undef, i32 %val, i32 0		; <<4 x i32>> [#uses=1]
	%tmp4 = insertelement <4 x i32> %tmp3, i32 %idx, i32 3		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp4
}
