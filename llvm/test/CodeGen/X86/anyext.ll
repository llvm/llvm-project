; RUN: llc < %s -march=x86-64 | grep movzbl | count 2

; Use movzbl to avoid partial-register updates.

define i32 @foo(i32 %p, i8 zeroext %x) nounwind {
  %q = trunc i32 %p to i8
  %r = udiv i8 %q, %x
  %s = zext i8 %r to i32
  %t = and i32 %s, 1
  ret i32 %t
}
define i32 @bar(i32 %p, i16 zeroext %x) nounwind {
  %q = trunc i32 %p to i16
  %r = udiv i16 %q, %x
  %s = zext i16 %r to i32
  %t = and i32 %s, 1
  ret i32 %t
}
