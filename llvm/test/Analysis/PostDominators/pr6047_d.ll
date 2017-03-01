; RUN: opt < %s -postdomtree -analyze | FileCheck %s
define internal void @f() {
entry:
  br i1 1, label %a, label %b

a:
br label %c

b:
br label %c

c:
  br i1 undef, label %bb35, label %bb3.i

bb3.i:
  br label %bb3.i

bb35.loopexit3:
  br label %bb35

bb35:
  ret void
}
; CHECK: Inorder PostDominator Tree: 
; CHECK-NEXT:   [1]  <<exit node>> {0,15}
; CHECK-NEXT:     [2] %bb35 {1,4}
; CHECK-NEXT:       [3] %bb35.loopexit3 {2,3}
; CHECK-NEXT:     [2] %c {5,12}
; CHECK-NEXT:       [3] %b {6,7}
; CHECK-NEXT:       [3] %entry {8,9}
; CHECK-NEXT:       [3] %a {10,11}
; CHECK-NEXT:     [2] %bb3.i {13,14}
