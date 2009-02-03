; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep bitrev %t1.s | count 1 
declare i32 @llvm.xcore.bitrev(i32)

define i32 @test(i32 %val) {
	%result = call i32 @llvm.xcore.bitrev(i32 %val)
	ret i32 %result
}
