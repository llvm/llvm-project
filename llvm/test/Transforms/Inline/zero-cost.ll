; RUN: opt -inline -S %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S %s | FileCheck %s

define void @f() {
entry:
  tail call void @g()
  unreachable

; CHECK-LABEL: @f
; CHECK-NOT: call
; CHECK: unreachable
}

define void @g() {
entry:
  unreachable
}

