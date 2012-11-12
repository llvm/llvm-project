; RUN: opt < %s -instcombine -S | grep ashr

define i32 @foo(i32 %x, i32 %y) {
  %a = and i32 %x, 7
  %b = and i32 %y, 7
  %c = mul i32 %a, %b
  %d = shl i32 %c, 26
  %e = ashr i32 %d, 26
  ret i32 %e
}
