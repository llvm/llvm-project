; RUN: opt < %s -postdomtree -analyze | FileCheck %s
define internal void @f() {
entry:
  br i1 undef, label %a, label %bb3.i

a:
  br i1 undef, label %bb35, label %bb3.i

bb3.i:
  br label %bb3.i


bb35.loopexit3:
  br label %bb35

bb35:
  ret void
}
; CHECK: Inorder PostDominator Tree: 
; CHECK-NEXT:   [1]  <<exit node>> {0,11}
; CHECK-NEXT:     [2] %bb35 {1,4}
; CHECK-NEXT:       [3] %bb35.loopexit3 {2,3}
; CHECK-NEXT:     [2] %a {5,6}
; CHECK-NEXT:     [2] %entry {7,8}
; CHECK-NEXT:     [2] %bb3.i {9,10}
