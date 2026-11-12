; RUN: opt < %s -passes='print<postdomtree>' 2>&1 | FileCheck %s
define internal void @f(i1 %arg) {
entry:
  br i1 %arg, label %a, label %bb3.i

a:
  br i1 %arg, label %bb35, label %bb3.i

bb3.i:
  br label %bb3.i


bb35.loopexit3:
  br label %bb35

bb35:
  ret void
}
; CHECK: Inorder PostDominator Tree:
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %bb35
; CHECK-NEXT:       [3] %bb35.loopexit3
; CHECK-NEXT:     [2] %a
; CHECK-NEXT:     [2] %entry
; CHECK-NEXT:     [2] %bb3.i
