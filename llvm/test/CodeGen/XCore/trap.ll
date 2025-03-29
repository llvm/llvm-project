; RUN: llc < %s -mtriple=xcore | FileCheck %s
define i32 @test() noreturn nounwind  {
entry:
; CHECK-LABEL: test:
; CHECK: ldc
; CHECK: ecallf
	tail call void @llvm.trap( )
	unreachable
}

declare void @llvm.trap() nounwind 

