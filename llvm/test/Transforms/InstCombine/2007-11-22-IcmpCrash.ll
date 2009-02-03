; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR1817

define i1 @test1(i32 %X) {
	%A = icmp slt i32 %X, 10
	%B = icmp ult i32 %X, 10
	%C = and i1 %A, %B
	ret i1 %C
}

define i1 @test2(i32 %X) {
	%A = icmp slt i32 %X, 10
	%B = icmp ult i32 %X, 10
	%C = or i1 %A, %B
	ret i1 %C
}
