; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi | FileCheck %s
; PR4659
; PR4682

define hidden i32 @__gcov_execlp(i8* %path, i8* %arg, ...) nounwind {
entry:
; CHECK: __gcov_execlp:
; CHECK: sub sp, #8
; CHECK: push
; CHECK: add r7, sp, #4
; CHECK: sub.w r4, r7, #4
; CHECK: mov sp, r4
; CHECK-NOT: mov sp, r7
; CHECK: add sp, #8
	call void @__gcov_flush() nounwind
	br i1 undef, label %bb5, label %bb

bb:		; preds = %bb, %entry
	br i1 undef, label %bb5, label %bb

bb5:		; preds = %bb, %entry
	%0 = alloca i8*, i32 undef, align 4		; <i8**> [#uses=1]
	%1 = call i32 @execvp(i8* %path, i8** %0) nounwind		; <i32> [#uses=1]
	ret i32 %1
}

declare hidden void @__gcov_flush()

declare i32 @execvp(i8*, i8**) nounwind
