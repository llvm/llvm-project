; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: grep movlhps %t | count 1
; RUN: grep movss %t | count 1
; RUN: grep pshufd %t | count 1
; RUN: grep pshuflw %t | count 1
; RUN: grep pshufhw %t | count 1

define <8 x i16> @t1(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 8, i32 9, i32 0, i32 1, i32 10, i32 11, i32 2, i32 3 >
	ret <8 x i16> %tmp
}

define <8 x i16> @t2(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 8, i32 9, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp
}

define <8 x i16> @t3(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 0, i32 0, i32 3, i32 2, i32 4, i32 6, i32 4, i32 7 >
	ret <8 x i16> %tmp
}
