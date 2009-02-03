; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "ecallf" %t1.s | count 1
; RUN: grep "ldc" %t1.s | count 1
define i32 @test() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

declare void @llvm.trap() nounwind 

