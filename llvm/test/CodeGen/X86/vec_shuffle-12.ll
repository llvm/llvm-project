; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: not grep punpck %t
; RUN: grep pextrw %t | count 4
; RUN: grep pinsrw %t | count 6
; RUN: grep pshuflw %t | count 3
; RUN: grep pshufhw %t | count 2

define <8 x i16> @t1(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp3
}

define <8 x i16> @t2(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 9, i32 1, i32 2, i32 9, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp
}

define <8 x i16> @t3(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %A, <8 x i32> < i32 8, i32 3, i32 2, i32 13, i32 7, i32 6, i32 5, i32 4 >
	ret <8 x i16> %tmp
}

define <8 x i16> @t4(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 0, i32 7, i32 2, i32 3, i32 1, i32 5, i32 6, i32 5 >
	ret <8 x i16> %tmp
}
