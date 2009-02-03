; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp eq i32 %In, 15}
; PR2479
; (See also PR1800.)

define i1 @test(i32 %In) {
	%c1 = icmp ugt i32 %In, 13
	%c2 = icmp eq i32 %In, 15
	%V = and i1 %c1, %c2
	ret i1 %V
}

