; RUN: llc < %s -march=sparc
; PR 1557

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @set_fast_math, ptr null } ]		; <[1 x { i32, void ()*, i8* }]*> [#uses=0]

define internal void @set_fast_math() nounwind {
entry:
	%fsr = alloca i32		; <i32*> [#uses=4]
	call void asm "st %fsr, $0", "=*m"(ptr elementtype(i32) %fsr) nounwind
	%0 = load i32, ptr %fsr, align 4		; <i32> [#uses=1]
	%1 = or i32 %0, 4194304		; <i32> [#uses=1]
	store i32 %1, ptr %fsr, align 4
	call void asm sideeffect "ld $0, %fsr", "*m"(ptr elementtype(i32) %fsr) nounwind
	ret void
}
