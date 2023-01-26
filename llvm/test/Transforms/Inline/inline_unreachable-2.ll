; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

; CHECK-LABEL: caller
; CHECK: call void @callee
define void @caller(i32 %a, i1 %b) #0 {
  call void @callee(i32 %a, i1 %b)
  unreachable
}

define void @callee(i32 %a, i1 %b) {
  call void @extern()
  call void asm sideeffect "", ""()
  br i1 %b, label %bb1, label %bb2
bb1:
  call void asm sideeffect "", ""()
  ret void
bb2:
  call void asm sideeffect "", ""()
  ret void
}

declare void @extern()
