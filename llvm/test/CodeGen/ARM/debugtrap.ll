; This test ensures the @llvm.debugtrap() call is not removed when generating
; the 'pop' instruction to restore the callee saved registers on ARM.

; RUN: llc < %s -mtriple=armv4 -O0 -filetype=asm | FileCheck --check-prefixes=CHECK,V4 %s
; RUN: llc < %s -mtriple=armv5 -O0 -filetype=asm | FileCheck --check-prefixes=CHECK,V5 %s
; RUN: llc < %s -mtriple=thumbv4 -O0 -filetype=asm | FileCheck --check-prefixes=CHECK,V4 %s
; RUN: llc < %s -mtriple=thumbv5 -O0 -filetype=asm | FileCheck --check-prefixes=CHECK,V5 %s

declare void @llvm.debugtrap() nounwind
declare void @foo() nounwind

define void @test() nounwind {
entry:
  ; CHECK: bl foo
  ; V4-NEXT: udf #254
  ; V5-NEXT: bkpt #0
  ; CHECK-NEXT: pop
  call void @foo()
  call void @llvm.debugtrap()
  ret void
}
