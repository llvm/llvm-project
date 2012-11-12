; RUN: llc < %s -march=x86 | grep inc
; RUN: llc < %s -march=x86 | grep add | grep 4

define i32 @test(i32 %X) nounwind {
entry:
	%0 = add i32 %X, 1
	ret i32 %0
}

define i32 @test2(i32 %X) nounwind {
entry:
	%0 = add i32 %X, 4
	ret i32 %0
}
