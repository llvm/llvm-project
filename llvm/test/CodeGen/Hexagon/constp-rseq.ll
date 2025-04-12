; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: cmp
; Make sure that the result is not a compile-time constant.

define i64 @foo(i32 %x) {
entry:
  %c = icmp slt i32 %x, 17
  br i1 %c, label %b1, label %b2
b1:
  br label %b2
b2:
  %p = phi i32 [ 1, %entry ], [ 0, %b1 ]
  %q = sub i32 %x, %x
  %y = zext i32 %q to i64
  %u = shl i64 %y, 32
  %v = zext i32 %p to i64
  %w = or i64 %u, %v
  ret i64 %w
}
