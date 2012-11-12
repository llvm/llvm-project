; RUN: llc < %s -march=xcore | FileCheck %s
define i32 @test() noreturn nounwind  {
entry:
; CHECK: test:
; CHECK: ldc
; CHECK: ecallf
	tail call void @llvm.trap( )
	unreachable
}

declare void @llvm.trap() nounwind 

