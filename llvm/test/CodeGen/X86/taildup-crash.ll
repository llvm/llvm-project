; RUN: llc -o - %s | FileCheck %s
target triple = "x86_64--"

; Make sure we do not crash in tail duplication when finding no successor of a
; block.
; CHECK-LABEL: func:
; CHECK: testb
; CHECK: je
; CHECK: retq
; CHECK: jmp
define hidden void @func() {
entry:
  br i1 undef, label %for.cond.cleanup, label %while.cond.preheader

while.cond.preheader:
  br label %while.cond

for.cond.cleanup:
  ret void

while.cond:
  %cmp.i202 = icmp eq ptr undef, undef
  br i1 %cmp.i202, label %while.cond.preheader, label %while.cond
}
