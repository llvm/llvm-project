; RUN: opt < %s -passes=slsr,gvn -S | FileCheck %s
; RUN: opt < %s -passes='slsr,gvn' -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

define void @slsr1(i32 %b, i32 %s) {
; CHECK-LABEL: @slsr1(
  ; foo(b * s);
  %mul0 = mul i32 %b, %s
; CHECK: mul i32
; CHECK-NOT: mul i32
  call void @foo(i32 %mul0)

  ; foo((b + 1) * s);
  %b1 = add i32 %b, 1
  %mul1 = mul i32 %b1, %s
  call void @foo(i32 %mul1)

  ; foo((b + 2) * s);
  %b2 = add i32 %b, 2
  %mul2 = mul i32 %b2, %s
  call void @foo(i32 %mul2)

  ret void
}

define void @non_canonicalized(i32 %b, i32 %s) {
; CHECK-LABEL: @non_canonicalized(
  ; foo(b * s);
  %mul0 = mul i32 %b, %s
; CHECK: mul i32
; CHECK-NOT: mul i32
  call void @foo(i32 %mul0)

  ; foo((1 + b) * s);
  %b1 = add i32 1, %b
  %mul1 = mul i32 %b1, %s
  call void @foo(i32 %mul1)

  ; foo((2 + b) * s);
  %b2 = add i32 2, %b
  %mul2 = mul i32 %b2, %s
  call void @foo(i32 %mul2)

  ret void
}

define void @or(i32 %a, i32 %s) {
  %b = shl i32 %a, 1
; CHECK-LABEL: @or(
  ; foo(b * s);
  %mul0 = mul i32 %b, %s
; CHECK: [[base:[^ ]+]] = mul i32
  call void @foo(i32 %mul0)

  ; foo((b | 1) * s);
  %b1 = or i32 %b, 1
  %mul1 = mul i32 %b1, %s
; CHECK: add i32 [[base]], %s
  call void @foo(i32 %mul1)

  ; foo((b | 2) * s);
  %b2 = or i32 %b, 2
  %mul2 = mul i32 %b2, %s
; CHECK: mul i32 %b2, %s
  call void @foo(i32 %mul2)

  ret void
}

; foo(a * b)
; foo((a + 1) * b)
; foo(a * (b + 1))
; foo((a + 1) * (b + 1))
define void @slsr2(i32 %a, i32 %b) {
; CHECK-LABEL: @slsr2(
  %a1 = add i32 %a, 1
  %b1 = add i32 %b, 1
  %mul0 = mul i32 %a, %b
; CHECK: mul i32
; CHECK-NOT: mul i32
  %mul1 = mul i32 %a1, %b
  %mul2 = mul i32 %a, %b1
  %mul3 = mul i32 %a1, %b1

  call void @foo(i32 %mul0)
  call void @foo(i32 %mul1)
  call void @foo(i32 %mul2)
  call void @foo(i32 %mul3)

  ret void
}

; The bump is a multiple of the stride.
;
; foo(b * s);
; foo((b + 2) * s);
; foo((b + 4) * s);
;   =>
; mul0 = b * s;
; bump = s * 2;
; mul1 = mul0 + bump; // GVN ensures mul1 and mul2 use the same bump.
; mul2 = mul1 + bump;
define void @slsr3(i32 %b, i32 %s) {
; CHECK-LABEL: @slsr3(
  %mul0 = mul i32 %b, %s
; CHECK: mul i32
  call void @foo(i32 %mul0)

  %b1 = add i32 %b, 2
  %mul1 = mul i32 %b1, %s
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = shl i32 %s, 1
; CHECK: %mul1 = add i32 %mul0, [[BUMP]]
  call void @foo(i32 %mul1)

  %b2 = add i32 %b, 4
  %mul2 = mul i32 %b2, %s
; CHECK: %mul2 = add i32 %mul1, [[BUMP]]
  call void @foo(i32 %mul2)

  ret void
}

; Do not rewrite a candidate if its potential basis does not dominate it.
;
; if (cond)
;   foo(a * b);
; foo((a + 1) * b);
define void @not_dominate(i1 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: @not_dominate(
entry:
  %a1 = add i32 %a, 1
  br i1 %cond, label %then, label %merge

then:
  %mul0 = mul i32 %a, %b
; CHECK: %mul0 = mul i32 %a, %b
  call void @foo(i32 %mul0)
  br label %merge

merge:
  %mul1 = mul i32 %a1, %b
; CHECK: %mul1 = mul i32 %a1, %b
  call void @foo(i32 %mul1)
  ret void
}

declare void @foo(i32)
