; RUN: llc -march=x86-64 < %s | not grep negq

; These sequences don't need neg instructions; they can be done with
; a single shift and sub each.

define i64 @foo(i64 %x, i64 %y, i64 %n) nounwind {
  %a = sub i64 0, %y
  %b = shl i64 %a, %n
  %c = add i64 %b, %x
  ret i64 %c
}
define i64 @boo(i64 %x, i64 %y, i64 %n) nounwind {
  %a = sub i64 0, %y
  %b = shl i64 %a, %n
  %c = add i64 %x, %b
  ret i64 %c
}
