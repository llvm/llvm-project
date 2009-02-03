; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp ugt}
; PR1107
; PR1940

define i1 @test(i8 %A, i8 %B) {
	%a = zext i8 %A to i32
	%b = zext i8 %B to i32
	%c = icmp sgt i32 %a, %b
	ret i1 %c
}
