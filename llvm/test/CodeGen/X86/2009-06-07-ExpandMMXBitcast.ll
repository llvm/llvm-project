; RUN: llc < %s -mtriple=x86_64-linux -mcpu=corei7 -mattr=+mmx | grep movd | count 2

define i64 @a(i32 %a, i32 %b) nounwind readnone {
entry:
	%0 = insertelement <2 x i32> undef, i32 %a, i32 0		; <<2 x i32>> [#uses=1]
	%1 = insertelement <2 x i32> %0, i32 %b, i32 1		; <<2 x i32>> [#uses=1]
	%conv = bitcast <2 x i32> %1 to i64		; <i64> [#uses=1]
	ret i64 %conv
}

