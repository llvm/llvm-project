; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -mtriple=i686-apple-darwin9 -stats -info-output-file - | grep asm-printer | grep 4
; PR2485

define <4 x i32> @t(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
	%shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> < i32 4, i32 0, i32 0, i32 0 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %shuffle
}
