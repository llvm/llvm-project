; RUN: opt < %s -instcombine -S | grep "xor"
; PR2389

define i1 @test(i1 %a, i1 %b) {
  %A = add i1 %a, %b
  ret i1 %A
}
