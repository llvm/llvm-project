; RUN: llc < %s -march=x86 -mcpu=core2 -o %t
; RUN: grep pshufb %t | count 1

define <8 x i16> @shuf3(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 0, i32 1, i32 undef, i32 undef, i32 3, i32 11, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
}
