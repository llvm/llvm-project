; RUN: llc < %s -mattr=-sse | FileCheck %s -check-prefix=NOSSE
; RUN: llc < %s | FileCheck %s -check-prefix=YESSSE
; PR3403
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.__va_list_tag = type { i32, i32, i8*, i8* }

; NOSSE-NOT: xmm
; YESSSE: xmm
define i32 @foo(float %a, i8* nocapture %fmt, ...) nounwind {
entry:
	%ap = alloca [1 x %struct.__va_list_tag], align 8		; <[1 x %struct.__va_list_tag]*> [#uses=4]
	%ap12 = bitcast [1 x %struct.__va_list_tag]* %ap to i8*		; <i8*> [#uses=2]
	call void @llvm.va_start(i8* %ap12)
	%0 = getelementptr [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 0		; <i32*> [#uses=2]
	%1 = load i32, i32* %0, align 8		; <i32> [#uses=3]
	%2 = icmp ult i32 %1, 48		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb3

bb:		; preds = %entry
	%3 = getelementptr [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 3		; <i8**> [#uses=1]
	%4 = load i8*, i8** %3, align 8		; <i8*> [#uses=1]
	%5 = inttoptr i32 %1 to i8*		; <i8*> [#uses=1]
	%6 = ptrtoint i8* %5 to i64		; <i64> [#uses=1]
	%ctg2 = getelementptr i8, i8* %4, i64 %6		; <i8*> [#uses=1]
	%7 = add i32 %1, 8		; <i32> [#uses=1]
	store i32 %7, i32* %0, align 8
	br label %bb4

bb3:		; preds = %entry
	%8 = getelementptr [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 2		; <i8**> [#uses=2]
	%9 = load i8*, i8** %8, align 8		; <i8*> [#uses=2]
	%10 = getelementptr i8, i8* %9, i64 8		; <i8*> [#uses=1]
	store i8* %10, i8** %8, align 8
	br label %bb4

bb4:		; preds = %bb3, %bb
	%addr.0.0 = phi i8* [ %ctg2, %bb ], [ %9, %bb3 ]		; <i8*> [#uses=1]
	%11 = bitcast i8* %addr.0.0 to i32*		; <i32*> [#uses=1]
	%12 = load i32, i32* %11, align 4		; <i32> [#uses=1]
	call void @llvm.va_end(i8* %ap12)
	ret i32 %12
}

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind
