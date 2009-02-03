; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {xor}
; PR2389

define i1 @test(i1 %a, i1 %b) {
  %A = add i1 %a, %b
  ret i1 %A
}
