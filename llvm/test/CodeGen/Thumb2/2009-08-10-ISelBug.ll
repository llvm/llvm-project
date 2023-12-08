; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+vfp2

define float @t1(i32 %v0) nounwind {
entry:
	store i32 undef, ptr undef, align 4
	%0 = load ptr, ptr undef, align 4		; <ptr> [#uses=1]
	%1 = load i8, ptr undef, align 1		; <i8> [#uses=1]
	%2 = zext i8 %1 to i32		; <i32> [#uses=1]
	%3 = getelementptr [4 x i8], ptr %0, i32 %v0, i32 0		; <ptr> [#uses=1]
	%4 = load i8, ptr %3, align 1		; <i8> [#uses=1]
	%5 = zext i8 %4 to i32		; <i32> [#uses=1]
	%6 = sub i32 %5, %2		; <i32> [#uses=1]
	%7 = sitofp i32 %6 to float		; <float> [#uses=1]
	ret float %7
}
