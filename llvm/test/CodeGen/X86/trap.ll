; RUN: llc < %s -march=x86 -mcpu=yonah | FileCheck %s

; CHECK: test0:
; CHECK: ud2
define i32 @test0() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

; CHECK: test1:
; CHECK: int3
define i32 @test1() noreturn nounwind  {
entry:
	tail call void @llvm.debugtrap( )
	unreachable
}

declare void @llvm.trap() nounwind 
declare void @llvm.debugtrap() nounwind 

