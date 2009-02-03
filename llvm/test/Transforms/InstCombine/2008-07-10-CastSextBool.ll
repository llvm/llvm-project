; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {%C = xor i1 %A, true}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i1 false}
; PR2539

define i1 @test1(i1 %A) {
	%B = zext i1 %A to i32
	%C = icmp slt i32 %B, 1
	ret i1 %C
}


define i1 @test2(i1 zeroext  %b) {
entry:
	%cmptmp = icmp slt i1 %b, true		; <i1> [#uses=1]
	ret i1 %cmptmp
}

