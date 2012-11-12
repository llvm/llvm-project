; RUN: llc < %s -mtriple=thumbv7-apple-ios -mattr=+vfp2,+thumb2 | FileCheck %s
; rdar://7076238

@"\01LC" = external constant [36 x i8], align 1		; <[36 x i8]*> [#uses=1]

define i32 @t(i32, ...) nounwind {
entry:
; CHECK: t:
; CHECK: add r7, sp, #12
	%1 = load i8** undef, align 4		; <i8*> [#uses=3]
	%2 = getelementptr i8* %1, i32 4		; <i8*> [#uses=1]
	%3 = getelementptr i8* %1, i32 8		; <i8*> [#uses=1]
	%4 = bitcast i8* %2 to i32*		; <i32*> [#uses=1]
	%5 = load i32* %4, align 4		; <i32> [#uses=1]
	%6 = trunc i32 %5 to i8		; <i8> [#uses=1]
	%7 = getelementptr i8* %1, i32 12		; <i8*> [#uses=1]
	%8 = bitcast i8* %3 to i32*		; <i32*> [#uses=1]
	%9 = load i32* %8, align 4		; <i32> [#uses=1]
	%10 = trunc i32 %9 to i16		; <i16> [#uses=1]
	%11 = bitcast i8* %7 to i32*		; <i32*> [#uses=1]
	%12 = load i32* %11, align 4		; <i32> [#uses=1]
	%13 = trunc i32 %12 to i16		; <i16> [#uses=1]
	%14 = load i32* undef, align 4		; <i32> [#uses=2]
	%15 = sext i8 %6 to i32		; <i32> [#uses=2]
	%16 = sext i16 %10 to i32		; <i32> [#uses=2]
	%17 = sext i16 %13 to i32		; <i32> [#uses=2]
	%18 = call  i32 (i8*, ...)* @printf(i8* getelementptr ([36 x i8]* @"\01LC", i32 0, i32 0), i32 -128, i32 0, i32 %15, i32 %16, i32 %17, i32 0, i32 %14) nounwind		; <i32> [#uses=0]
	%19 = add i32 0, %15		; <i32> [#uses=1]
	%20 = add i32 %19, %16		; <i32> [#uses=1]
	%21 = add i32 %20, %14		; <i32> [#uses=1]
	%22 = add i32 %21, %17		; <i32> [#uses=1]
	%23 = add i32 %22, 0		; <i32> [#uses=1]
	ret i32 %23
}

declare i32 @printf(i8* nocapture, ...) nounwind
