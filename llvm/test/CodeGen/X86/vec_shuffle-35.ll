; RUN: llc < %s -march=x86 -mcpu=yonah -stack-alignment=16 -o %t
; RUN: grep pextrw %t | count 12
; RUN: grep pinsrw %t | count 13
; RUN: grep rolw %t | count 13
; RUN: not grep esp %t
; RUN: not grep ebp %t
; RUN: llc < %s -march=x86 -mcpu=core2 -stack-alignment=16 -o %t
; RUN: grep pshufb %t | count 3

define <16 x i8> @shuf1(<16 x i8> %T0) nounwind readnone {
entry:
	%tmp8 = shufflevector <16 x i8> %T0, <16 x i8> undef, <16 x i32> < i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 12, i32 13, i32 15 , i32 14 >
	ret <16 x i8> %tmp8
}

define <16 x i8> @shuf2(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
entry:
	%tmp8 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> < i32 undef, i32 undef, i32 3, i32 2, i32 17, i32 16, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 12, i32 13, i32 15 , i32 14 >
	ret <16 x i8> %tmp8
}
