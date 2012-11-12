; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=sse41 -o %t
; RUN: grep punpcklbw %t | count 16
; RUN: grep punpckhbw %t | count 16
; RUN: grep "pshufd" %t | count 16

; Should generate with pshufd with masks $0, $85, $170, $255 (each mask is used 4 times)

; Splat test for v16i8
define <16 x i8 > @shuf_16i8_0(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 0, i32 undef, i32 undef, i32 0, i32 undef, i32 0, i32 0 , i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_1(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef, i32 undef, i32 undef, i32 undef  >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_2(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 2, i32 undef, i32 undef, i32 2, i32 undef, i32 2, i32 2 , i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_3(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 3, i32 undef, i32 undef, i32 3, i32 undef, i32 3, i32 3 , i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3 >
	ret <16 x i8 > %tmp6
}


define <16 x i8 > @shuf_16i8_4(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 4, i32 undef, i32 undef, i32 undef, i32 4, i32 undef, i32 undef , i32 undef, i32 undef, i32 undef, i32 undef , i32 undef, i32 undef, i32 undef, i32 undef , i32 undef  >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_5(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 5, i32 undef, i32 undef, i32 5, i32 undef, i32 5, i32 5 , i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_6(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 6, i32 undef, i32 undef, i32 6, i32 undef, i32 6, i32 6 , i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_7(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 7, i32 undef, i32 undef, i32 7, i32 undef, i32 undef, i32 undef , i32 undef, i32 undef, i32 undef, i32 undef , i32 undef , i32 undef, i32 undef, i32 undef , i32 undef  >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_8(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 8, i32 undef, i32 undef, i32 8, i32 undef, i32 8, i32 8 , i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_9(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 9, i32 undef, i32 undef, i32 9, i32 undef, i32 9, i32 9 , i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_10(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 10, i32 undef, i32 undef, i32 10, i32 undef, i32 10, i32 10 , i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_11(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 11, i32 undef, i32 undef, i32 11, i32 undef, i32 11, i32 11 , i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_12(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 12, i32 undef, i32 undef, i32 12, i32 undef, i32 12, i32 12 , i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_13(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 13, i32 undef, i32 undef, i32 13, i32 undef, i32 13, i32 13 , i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_14(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 14, i32 undef, i32 undef, i32 14, i32 undef, i32 14, i32 14 , i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14 >
	ret <16 x i8 > %tmp6
}

define <16 x i8 > @shuf_16i8_15(<16 x i8 > %T0, <16 x i8 > %T1) nounwind readnone {
entry:
	%tmp6 = shufflevector <16 x i8 > %T0, <16 x i8 > %T1, <16 x i32> < i32 15, i32 undef, i32 undef, i32 15, i32 undef, i32 15, i32 15 , i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15 >
	ret <16 x i8 > %tmp6
}
