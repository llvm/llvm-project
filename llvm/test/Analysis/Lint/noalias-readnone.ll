; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck --allow-empty %s

declare void @foo1(ptr noalias, ptr readnone)

define void @test1(ptr %a) {
entry:
  call void @foo1(ptr %a, ptr %a)
  ret void
}

; Lint should not complain about passing %a to both arguments even if one is
; noalias, since the second argument is readnone.
; CHECK-NOT: Unusual: noalias argument aliases another argument
; CHECK-NOT: call void @foo1(ptr %a, ptr %a)
