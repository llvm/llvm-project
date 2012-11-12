; RUN: llc < %s -march=x86 -mcpu=core2 | grep pshufb | count 2

define <8 x i16> @shuf2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp8
}
