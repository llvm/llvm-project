; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S -mtriple=x86_64 < %s | FileCheck %s

; Test that SplitIndirectBrCriticalEdges does not crash when a predecessor
; block has a conditional branch with both targets pointing to the same
; destination (producing duplicate entries in the predecessor list).

; CHECK-LABEL: @duplicate_pred_condbr
; CHECK: indirectbr ptr %addr, [label %target, label %cond.bb]
; CHECK: cond.bb:
; CHECK-NEXT: br label %.split
; CHECK: .split:
; CHECK-NEXT: %merge = phi i32
; CHECK-NEXT: ret i32 %merge

define i32 @duplicate_pred_condbr(ptr %addr, i1 %cond) {
entry:
  indirectbr ptr %addr, [label %target, label %cond.bb]

cond.bb:
  br i1 %cond, label %target, label %target

target:
  %result = phi i32 [ 0, %entry ], [ 1, %cond.bb ], [ 1, %cond.bb ]
  ret i32 %result
}
