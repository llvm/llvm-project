; Test SystemZInstrInfo::analyzeBranch and SystemZInstrInfo::insertBranch.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @foo() noreturn

; Check a case where a separate branch is needed and where the original
; order should be reversed.
define i32 @f1(i32 %a, ptr %bptr) {
; CHECK-LABEL: f1:
; CHECK: cl %r2, 0(%r3)
; CHECK: jl .L[[LABEL:.*]]
; CHECK: br %r14
; CHECK: .L[[LABEL]]:
; CHECK: brasl %r14, foo@PLT
entry:
  %b = load i32, ptr %bptr
  %cmp = icmp ult i32 %a, %b
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}

; Same again with a fused compare and branch.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: cije %r2, 0, .L[[LABEL:.*]]
; CHECK: br %r14
; CHECK: .L[[LABEL]]:
; CHECK: brasl %r14, foo@PLT
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %callit, label %return

callit:
  call void @foo()
  unreachable

return:
  ret i32 1
}
