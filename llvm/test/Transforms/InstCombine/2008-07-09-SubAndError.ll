; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {sub i32 0}
; PR2330

define i32 @foo(i32 %a) nounwind {
entry:
  %A = sub i32 5, %a
  %B = and i32 %A, 2
  ret i32 %B
}
