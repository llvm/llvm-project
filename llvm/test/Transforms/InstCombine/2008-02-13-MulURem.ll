; RUN: opt < %s -instcombine -S | grep rem
; PR1933

define i32 @fold(i32 %a) {
  %s = mul i32 %a, 3
  %c = urem i32 %s, 3
  ret i32 %c
}
