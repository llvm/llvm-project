; RUN: llc < %s
; PR3610
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-s0:0:64-f80:32:32"
target triple = "arm-elf"

define i32 @main(ptr) nounwind {
entry:
	%ap = alloca ptr		; <ptr> [#uses=2]
	store ptr %0, ptr %ap
	%retval = alloca i32		; <ptr> [#uses=2]
	store i32 0, ptr %retval
	%tmp = alloca float		; <ptr> [#uses=1]
	%1 = va_arg ptr %ap, float		; <float> [#uses=1]
	store float %1, ptr %tmp
	br label %return

return:		; preds = %entry
	%2 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %2
}
