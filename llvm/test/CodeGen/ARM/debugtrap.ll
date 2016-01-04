; This test ensures the @llvm.debugtrap() call is not removed when generating
; the 'pop' instruction to restore the callee saved registers on ARM.

; RUN: llc < %s -mtriple=armv7 -O0 -filetype=asm | FileCheck %s 

declare void @llvm.debugtrap() nounwind
declare void @foo() nounwind

define void @test() nounwind {
entry:
  ; CHECK: bl foo
  ; CHECK-NEXT: pop
  ; CHECK-NEXT: trap
  call void @foo()
  call void @llvm.debugtrap()
  ret void
}
