; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

; 'bar' can be overridden at link-time, don't inline it.
define weak void @bar() {
; CHECK-LABEL: define weak void @bar()
entry:
  ret void
}

define void @foo() {
; CHECK-LABEL: define void @foo()
entry:
  tail call void @bar()
; CHECK: tail call void @bar()
  ret void
}

