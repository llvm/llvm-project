;; RUN: opt -S < %s -passes=indvars | FileCheck %s

;; Check if IndVarSimplify understands !range metadata.

declare void @abort()

define i1 @iterate(i32* nocapture readonly %buffer) {
entry:
  %length = load i32, i32* %buffer, !range !0
  br label %loop.preheader

loop.preheader:
  br label %loop

loop:
  %idx = phi i32 [ %idx.inc, %loop.next ], [ 0, %loop.preheader ]
  %oob.pred = icmp slt i32 %idx, %length
  br i1 %oob.pred, label %loop.next, label %oob
; CHECK: br i1 true, label %loop.next, label %oob

loop.next:
  %idx.inc = add i32 %idx, 1
  %exit.pred = icmp slt i32 %idx.inc, %length
  br i1 %exit.pred, label %loop, label %abort.loopexit

abort.loopexit:
  br label %abort

abort:
  ret i1 false

oob:
  tail call void @abort()
  ret i1 false
}

!0 = !{i32 1, i32 100}
