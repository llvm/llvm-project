; RUN: opt < %s -instcombine -S | FileCheck %s
; PR2389

; CHECK: xor

define i1 @test(i1 %a, i1 %b) {
  %A = add i1 %a, %b
  ret i1 %A
}
