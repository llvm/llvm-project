; RUN: llc < %s -mtriple=thumbv7-apple-ios -mattr=+vfp2,+thumb2 | FileCheck %s
; rdar://7076238

@"\01LC" = external constant [36 x i8], align 1		; <ptr> [#uses=1]

define i32 @t(i32, ...) nounwind "frame-pointer"="all" {
entry:
; CHECK-LABEL: t:
; CHECK: add r7, sp, #12
	%1 = load ptr, ptr undef, align 4		; <ptr> [#uses=3]
	%2 = getelementptr i8, ptr %1, i32 4		; <ptr> [#uses=1]
	%3 = getelementptr i8, ptr %1, i32 8		; <ptr> [#uses=1]
	%4 = load i32, ptr %2, align 4		; <i32> [#uses=1]
	%5 = trunc i32 %4 to i8		; <i8> [#uses=1]
	%6 = getelementptr i8, ptr %1, i32 12		; <ptr> [#uses=1]
	%7 = load i32, ptr %3, align 4		; <i32> [#uses=1]
	%8 = trunc i32 %7 to i16		; <i16> [#uses=1]
	%9 = load i32, ptr %6, align 4		; <i32> [#uses=1]
	%10 = trunc i32 %9 to i16		; <i16> [#uses=1]
	%11 = load i32, ptr undef, align 4		; <i32> [#uses=2]
	%12 = sext i8 %5 to i32		; <i32> [#uses=2]
	%13 = sext i16 %8 to i32		; <i32> [#uses=2]
	%14 = sext i16 %10 to i32		; <i32> [#uses=2]
	%15 = call  i32 (ptr, ...) @printf(ptr @"\01LC", i32 -128, i32 0, i32 %12, i32 %13, i32 %14, i32 0, i32 %11) nounwind		; <i32> [#uses=0]
	%16 = add i32 0, %12		; <i32> [#uses=1]
	%17 = add i32 %16, %13		; <i32> [#uses=1]
	%18 = add i32 %17, %11		; <i32> [#uses=1]
	%19 = add i32 %18, %14		; <i32> [#uses=1]
	%20 = add i32 %19, 0		; <i32> [#uses=1]
	ret i32 %20
}

declare i32 @printf(ptr nocapture, ...) nounwind
