; RUN: opt < %s -passes='print<postdomtree>' 2>&1 | FileCheck %s
define internal void @f(i1 %arg) {
entry:
  br i1 1, label %a, label %b

a:
br label %c

b:
br label %c

c:
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
; CHECK-NEXT:     [2] %c
; CHECK-NEXT:       [3] %a
; CHECK-NEXT:       [3] %entry
; CHECK-NEXT:       [3] %b
; CHECK-NEXT:     [2] %bb3.i
