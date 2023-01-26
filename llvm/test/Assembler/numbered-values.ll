; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; RUN: verify-uselistorder %s
; PR2480

define i32 @test(i32 %X) nounwind {
entry:
	%X_addr = alloca i32		; <ptr> [#uses=2]
	%retval = alloca i32		; <ptr> [#uses=2]
	%0 = alloca i32		; <ptr>:0 [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %X, ptr %X_addr
	%1 = load i32, ptr %X_addr, align 4		; <i32>:1 [#uses=1]
	mul i32 %1, 4		; <i32>:2 [#uses=1]
	%3 = add i32 %2, 123		; <i32>:3 [#uses=1]
	store i32 %3, ptr %0, align 4
	ret i32 %3
}
