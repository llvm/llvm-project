; RUN: opt < %s -instcombine -S | FileCheck %s
define i32 @foo(i32 %x, i32 %y) {
; CHECK: @foo
  %add = add nsw i32 %y, %x
  %mul = mul nsw i32 %add, %y
  %square = mul nsw i32 %y, %y
  %res = sub i32 %mul, %square
  ret i32 %res
; CHECK-NEXT: mul i32 %x, %y
; CHECK-NEXT: ret i32
}

define i1 @bar(i64 %x, i64 %y) {
; CHECK: @bar
  %a = and i64 %y, %x
; CHECK: and
; CHECK-NOT: and
  %not = xor i64 %a, -1
  %b = and i64 %y, %not
  %r = icmp eq i64 %b, 0
  ret i1 %r
; CHECK: ret i1
}
