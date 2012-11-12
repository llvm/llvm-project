; RUN: opt < %s -scalar-evolution -analyze \
; RUN:   | grep "\-->  (zext" | count 2

define i32 @foo(i32 %x) {
  %n = and i32 %x, 255
  %y = xor i32 %n, 255
  ret i32 %y
}
