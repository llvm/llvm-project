; Do not remove the invoke!
;
; RUN: llvm-as < %s | opt -simplifycfg -disable-output

define i32 @test() {
	%A = invoke i32 @test( )
			to label %Ret unwind label %Ret2		; <i32> [#uses=1]
Ret:		; preds = %0
	ret i32 %A
Ret2:		; preds = %0
	ret i32 undef
}

