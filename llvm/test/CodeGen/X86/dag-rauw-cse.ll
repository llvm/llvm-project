; RUN: llvm-as < %s | llc -march=x86 | grep {orl	\$1}
; PR3018

define i32 @test(i32 %A) nounwind {
  %B = or i32 %A, 1
  %C = or i32 %B, 1
  %D = and i32 %C, 7057
  ret i32 %D
}
