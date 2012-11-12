; RUN: llc < %s
; PR4057
define void @test_cast_float_to_char(i8* %result) nounwind {
entry:
	%result_addr = alloca i8*		; <i8**> [#uses=2]
	%test = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8* %result, i8** %result_addr
	store float 0x40B2AFA160000000, float* %test, align 4
	%0 = load float* %test, align 4		; <float> [#uses=1]
	%1 = fptosi float %0 to i8		; <i8> [#uses=1]
	%2 = load i8** %result_addr, align 4		; <i8*> [#uses=1]
	store i8 %1, i8* %2, align 1
	br label %return

return:		; preds = %entry
	ret void
}
