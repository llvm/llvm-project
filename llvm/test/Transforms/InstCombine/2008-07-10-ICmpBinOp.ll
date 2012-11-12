; RUN: opt < %s -instcombine -S | not grep add
; RUN: opt < %s -instcombine -S | not grep mul
; PR2330

define i1 @f(i32 %x, i32 %y) nounwind {
entry:
  %A = add i32 %x, 5
  %B = add i32 %y, 5
  %C = icmp eq i32 %A, %B
  ret i1 %C
}

define i1 @g(i32 %x, i32 %y) nounwind {
entry:
  %A = mul i32 %x, 5
  %B = mul i32 %y, 5
  %C = icmp eq i32 %A, %B
  ret i1 %C
}
