; RUN: opt -scoped-noalias -basicaa -gvn -S < %s | FileCheck %s

define i32 @test1(i32* %p, i32* %q) {
; CHECK-LABEL: @test1(i32* %p, i32* %q)
; CHECK: load i32, i32* %p
; CHECK-NOT: noalias
; CHECK: %c = add i32 %a, %a
  %a = load i32, i32* %p, !noalias !0
  %b = load i32, i32* %p
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(i32* %p, i32* %q) {
; CHECK-LABEL: @test2(i32* %p, i32* %q)
; CHECK: load i32, i32* %p, !alias.scope !0
; CHECK: %c = add i32 %a, %a
  %a = load i32, i32* %p, !alias.scope !0
  %b = load i32, i32* %p, !alias.scope !0
  %c = add i32 %a, %b
  ret i32 %c
}

; FIXME: In this case we can do better than intersecting the scopes, and can
; concatenate them instead. Both loads are in the same basic block, the first
; makes the second safe to speculatively execute, and there are no calls that may
; throw in between.
define i32 @test3(i32* %p, i32* %q) {
; CHECK-LABEL: @test3(i32* %p, i32* %q)
; CHECK: load i32, i32* %p, !alias.scope !1
; CHECK: %c = add i32 %a, %a
  %a = load i32, i32* %p, !alias.scope !1
  %b = load i32, i32* %p, !alias.scope !2
  %c = add i32 %a, %b
  ret i32 %c
}

declare i32 @foo(i32*) readonly

!0 = !{!0}
!1 = !{!1}
!2 = !{!0, !1}

