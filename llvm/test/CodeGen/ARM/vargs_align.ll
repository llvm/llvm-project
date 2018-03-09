; RUN: llc < %s -mtriple=armv7-linux-gnueabihf | FileCheck %s -check-prefix=EABI
; RUN: llc < %s -mtriple=arm-linux-gnu | FileCheck %s -check-prefix=OABI

define i32 @f(i32 %a, ...) {
entry:
	%a_addr = alloca i32		; <i32*> [#uses=1]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	store i32 %a, i32* %a_addr
	store i32 0, i32* %tmp
	%tmp1 = load i32, i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp1, i32* %retval
	call void @llvm.va_start(i8* null)
	call void asm sideeffect "", "~{d8}"()
	br label %return

return:		; preds = %entry
	%retval2 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval2
; EABI: add sp, sp, #16
; EABI: vpop {d8}
; EABI: add sp, sp, #4
; EABI: add sp, sp, #12

; OABI: add sp, sp, #12
; OABI: add sp, sp, #12
}

declare void @llvm.va_start(i8*) nounwind
