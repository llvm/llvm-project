; RUN: llc < %s
; PR3610
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-s0:0:64-f80:32:32"
target triple = "arm-elf"

define i32 @main(i8*) nounwind {
entry:
	%ap = alloca i8*		; <i8**> [#uses=2]
	store i8* %0, i8** %ap
	%retval = alloca i32		; <i32*> [#uses=2]
	store i32 0, i32* %retval
	%tmp = alloca float		; <float*> [#uses=1]
	%1 = va_arg i8** %ap, float		; <float> [#uses=1]
	store float %1, float* %tmp
	br label %return

return:		; preds = %entry
	%2 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %2
}
