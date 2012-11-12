; RUN: llc < %s -march=x86 | FileCheck %s
; PR3018

define i32 @test(i32 %A) nounwind {
; CHECK: test:
; CHECK-NOT: ret
; CHECK: orl $1
; CHECK: ret
  %B = or i32 %A, 1
  %C = or i32 %B, 1
  %D = and i32 %C, 7057
  ret i32 %D
}
