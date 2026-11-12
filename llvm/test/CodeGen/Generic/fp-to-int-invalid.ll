; RUN: llc < %s
; PR4057
define void @test_cast_float_to_char(ptr %result) nounwind {
entry:
	%result_addr = alloca ptr		; <ptr> [#uses=2]
	%test = alloca float		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %result, ptr %result_addr
	store float 0x40B2AFA160000000, ptr %test, align 4
	%0 = load float, ptr %test, align 4		; <float> [#uses=1]
	%1 = fptosi float %0 to i8		; <i8> [#uses=1]
	%2 = load ptr, ptr %result_addr, align 4		; <ptr> [#uses=1]
	store i8 %1, ptr %2, align 1
	br label %return

return:		; preds = %entry
	ret void
}
