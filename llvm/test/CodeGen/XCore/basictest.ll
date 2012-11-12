; RUN: llc < %s -march=xcore

define i32 @test(i32 %X) {
	%tmp.1 = add i32 %X, 1
	ret i32 %tmp.1
}
