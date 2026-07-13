; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck --allow-empty %s

declare void @foo(ptr noalias, ptr noalias)

define void @test() {
entry:
  call void @foo(ptr null, ptr null)
  ret void
}

; Lint should not complain about passing null to both arguments if they are
; null, since noalias only applies if the argument is written to, which is not
; possible for a null pointer.
; CHECK-NOT: Unusual: noalias argument aliases another argument
