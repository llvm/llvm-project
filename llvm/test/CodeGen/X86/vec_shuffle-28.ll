; RUN: llc < %s -march=x86 -mcpu=core2 -o %t
; RUN: grep pshufb %t | count 1

; FIXME: this test has a superfluous punpcklqdq pre-pshufb currently.
;        Don't XFAIL it because it's still better than the previous code.

; Pack various elements via shuffles.
define <8 x i16> @shuf1(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp7 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 1, i32 8, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp7
}
