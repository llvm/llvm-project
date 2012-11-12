; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=sse41 -o %t
; RUN: grep punpcklwd %t | count 4
; RUN: grep punpckhwd %t | count 4
; RUN: grep "pshufd" %t | count 8

; Splat test for v8i16
; Should generate with pshufd with masks $0, $85, $170, $255 (each mask is used twice)
define <8 x i16> @shuf_8i16_0(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 0, i32 undef, i32 undef, i32 0, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_1(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 2, i32 undef, i32 undef, i32 2, i32 undef, i32 2, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_3(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 3, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_4(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 4, i32 undef, i32 undef, i32 undef, i32 4, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_5(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 5, i32 undef, i32 undef, i32 5, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}

define <8 x i16> @shuf_8i16_6(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 6, i32 6, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}


define <8 x i16> @shuf_8i16_7(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 7, i32 undef, i32 undef, i32 7, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp6
}
