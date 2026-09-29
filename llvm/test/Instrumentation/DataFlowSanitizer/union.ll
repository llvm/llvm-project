; RUN: opt < %s -passes=dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i32 0
@b = common global i32 0

; Check that we reuse unions where possible.

; CHECK-LABEL: @f.dfsan
define void @f(i32 %x, i32 %y) {
  ; CHECK: or i8
  %xay = add i32 %x, %y
  store i32 %xay, ptr @a
  ; CHECK-NOT: or i8
  %xmy = mul i32 %x, %y
  store i32 %xmy, ptr @b
  ret void
}

; In this case, we compute the unions on both sides because neither block
; dominates the other.

; CHECK-LABEL: @g.dfsan
define void @g(i1 %p, i32 %x, i32 %y) {
  br i1 %p, label %l1, label %l2

l1:
  ; CHECK: or i8
  %xay = add i32 %x, %y
  store i32 %xay, ptr @a
  br label %l3

l2:
  ; CHECK: or i8
  %xmy = mul i32 %x, %y
  store i32 %xmy, ptr @b
  br label %l3

l3:
  ret void
}

; In this case, we know that the label for %xayax subsumes the label for %xay.

; CHECK-LABEL: @h.dfsan
define i32 @h(i32 %x, i32 %y) {
  ; CHECK: or i8
  %xay = add i32 %x, %y
  ; CHECK-NOT: or i8
  %xayax = add i32 %xay, %x
  ret i32 %xayax
}
