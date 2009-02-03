; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; PR3080
define i64 @test(i64 %a) {
	%result = shl i64 %a, 1
	ret i64 %result
}
