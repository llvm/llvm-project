; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {xor }

; PR1253
define i1 @test0(i32 %A) {
	%B = xor i32 %A, -2147483648
	%C = icmp sgt i32 %B, -1
	ret i1 %C
}

define i1 @test1(i32 %A) {
	%B = xor i32 %A, 12345
	%C = icmp slt i32 %B, 0
	ret i1 %C
}

