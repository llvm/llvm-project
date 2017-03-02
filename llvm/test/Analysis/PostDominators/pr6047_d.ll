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
; CHECK: [4] %entry
