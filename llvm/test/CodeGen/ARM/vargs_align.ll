; RUN: llc < %s -mtriple=armv7-linux-gnueabihf | FileCheck %s -check-prefix=EABI
; RUN: llc < %s -mtriple=arm-linux-gnu | FileCheck %s -check-prefix=OABI

define i32 @f(i32 %a, ...) {
entry:
	%a_addr = alloca i32		; <ptr> [#uses=1]
	%retval = alloca i32, align 4		; <ptr> [#uses=2]
	%tmp = alloca i32, align 4		; <ptr> [#uses=2]
	store i32 %a, ptr %a_addr
	store i32 0, ptr %tmp
	%tmp1 = load i32, ptr %tmp		; <i32> [#uses=1]
	store i32 %tmp1, ptr %retval
	call void @llvm.va_start(ptr null)
	call void asm sideeffect "", "~{d8}"()
	br label %return

return:		; preds = %entry
	%retval2 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval2
; EABI: add sp, sp, #16
; EABI: vpop {d8}
; EABI: add sp, sp, #4
; EABI: add sp, sp, #12

; OABI: add sp, sp, #24
}

declare void @llvm.va_start(ptr) nounwind
