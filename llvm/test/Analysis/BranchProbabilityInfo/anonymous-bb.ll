; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

define void @fct() {
; CHECK-LABEL: fct
entry:
  br label %0
0:
  br label %1
1:
  ret void
}

; CHECK: edge %entry -> %0
; CHECK: edge %0 -> %1

define void @fct2() {
; CHECK-LABEL: fct2
  br label %1
1:
  br label %2
2:
  ret void
}
; CHECK: edge %0 -> %1
; CHECK: edge %1 -> %2