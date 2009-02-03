; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep {select i1 %ispos}
; PR2337

define i32 @test(i32 %x) {
entry:
	%call = call i32 @abs( i32 %x )		; <i32> [#uses=1]
	ret i32 %call
}

declare i32 @abs(i32)

