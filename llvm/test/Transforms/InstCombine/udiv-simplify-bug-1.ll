; RUN: opt < %s -instcombine -S > %t1.ll
; RUN: grep udiv %t1.ll | count 2
; RUN: grep zext %t1.ll | count 2
; PR2274

; The udiv instructions shouldn't be optimized away, and the
; sext instructions should be optimized to zext.

define i64 @bar(i32 %x, i32 %g) nounwind {
  %y = lshr i32 %x, 30
  %r = udiv i32 %y, %g
  %z = sext i32 %r to i64
  ret i64 %z
}
define i64 @qux(i32 %x, i32 %v) nounwind {
  %y = lshr i32 %x, 31
  %r = udiv i32 %y, %v
  %z = sext i32 %r to i64
  ret i64 %z
}
