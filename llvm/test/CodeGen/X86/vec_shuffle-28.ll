; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 -o %t -f
; RUN: grep punpcklwd %t | count 1
; RUN: grep pextrw %t | count 6
; RUN: grep pinsrw %t | count 8


; Pack various elements via shuffles.
define <8 x i16> @shuf1(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp7 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 1, i32 8, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp7
}


define <8 x i16> @shuf2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp8
}


define <8 x i16> @shuf3(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 0, i32 1, i32 undef, i32 undef, i32 3, i32 11, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
}


define <8 x i16> @shuf4(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 8, i32 9, i32 undef, i32 undef, i32 11, i32 3, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
}
