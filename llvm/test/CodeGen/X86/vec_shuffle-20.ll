; RUN: llc < %s -o /dev/null -march=x86 -mcpu=corei7 -mtriple=i686-apple-darwin9 -stats -info-output-file - | grep asm-printer | grep 3

define <4 x float> @func(<4 x float> %fp0, <4 x float> %fp1) nounwind  {
entry:
	shufflevector <4 x float> %fp0, <4 x float> %fp1, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:0 [#uses=1]
	ret <4 x float> %0
}
