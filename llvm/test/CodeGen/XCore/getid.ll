; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "get r11, id" %t1.s | count 1 
declare i32 @llvm.xcore.getid()

define i32 @test() {
	%result = call i32 @llvm.xcore.getid()
	ret i32 %result
}
