; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi -arm-atomic-cfg-tidy=0 | FileCheck %s
; PR4659
; PR4682

define hidden i32 @__gcov_execlp(ptr %path, ptr %arg, ...) nounwind {
entry:
; CHECK-LABEL: __gcov_execlp:
; CHECK: sub sp, #8
; CHECK: push
; CHECK: add r7, sp, #8
; CHECK: sub.w r4, r7, #8
; CHECK: mov sp, r4
; CHECK-NOT: mov sp, r7
; CHECK: add sp, #8
	call void @__gcov_flush() nounwind
	call void @llvm.va_start(ptr null)
	br i1 undef, label %bb5, label %bb

bb:		; preds = %bb, %entry
	br i1 undef, label %bb5, label %bb

bb5:		; preds = %bb, %entry
	%0 = alloca ptr, i32 undef, align 4		; <ptr> [#uses=1]
	%1 = call i32 @execvp(ptr %path, ptr %0) nounwind		; <i32> [#uses=1]
	ret i32 %1
}

declare hidden void @__gcov_flush()

declare i32 @execvp(ptr, ptr) nounwind

declare void @llvm.va_start(ptr) nounwind
