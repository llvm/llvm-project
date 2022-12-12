;; RUN: opt -S < %s -passes=indvars | FileCheck %s
; RUN: opt -lcssa -loop-simplify -S < %s | opt -S -passes='require<targetir>,require<scalar-evolution>,require<domtree>,loop(indvars)'

;; Check if llvm can narrow !range metadata based on loop entry
;; predicates.

declare void @abort()

define i1 @bounded_below_slt(i32* nocapture readonly %buffer) {
; CHECK-LABEL: bounded_below_slt
entry:
  %length = load i32, i32* %buffer, !range !0
  %entry.pred = icmp eq i32 %length, 0
  br i1 %entry.pred, label %abort, label %loop.preheader

loop.preheader:
  br label %loop

loop:
; CHECK: loop
  %idx = phi i32 [ %idx.inc, %loop.next ], [ 0, %loop.preheader ]
  %oob.pred = icmp slt i32 %idx, %length
  br i1 %oob.pred, label %loop.next, label %oob
; CHECK: br i1 true, label %loop.next, label %oob

loop.next:
; CHECK: loop.next
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

define i1 @bounded_below_sle(i32* nocapture readonly %buffer) {
; CHECK-LABEL: bounded_below_sle
entry:
  %length = load i32, i32* %buffer, !range !0
  %entry.pred = icmp eq i32 %length, 0
  br i1 %entry.pred, label %abort, label %loop.preheader

loop.preheader:
  br label %loop

loop:
; CHECK: loop
  %idx = phi i32 [ %idx.inc, %loop.next ], [ 0, %loop.preheader ]
  %oob.pred = icmp sle i32 %idx, %length
  br i1 %oob.pred, label %loop.next, label %oob
; CHECK: br i1 true, label %loop.next, label %oob

loop.next:
; CHECK: loop.next
  %idx.inc = add i32 %idx, 1
  %exit.pred = icmp sle i32 %idx.inc, %length
  br i1 %exit.pred, label %loop, label %abort.loopexit

abort.loopexit:
  br label %abort

abort:
  ret i1 false

oob:
  tail call void @abort()
  ret i1 false
}

;; Assert that we're not making an incorrect transform.

declare i32 @check(i8*)

define void @NoChange() {
; CHECK-LABEL: NoChange
entry:
  br label %loop.begin

loop.begin:
; CHECK: loop.begin:
  %i.01 = phi i64 [ 2, %entry ], [ %add, %loop.end ]
  %cmp = icmp ugt i64 %i.01, 1
; CHECK: %cmp = icmp ugt i64 %i.01, 1
  br i1 %cmp, label %loop, label %loop.end

loop:
; CHECK: loop
  %.sum = add i64 %i.01, -2
  %v = getelementptr inbounds i8, i8* null, i64 %.sum
  %r = tail call i32 @check(i8* %v)
  %c = icmp eq i32 %r, 0
  br i1 %c, label %loop.end, label %abort.now

abort.now:
  tail call void @abort()
  unreachable

loop.end:
  %add = add i64 %i.01, -1
  %eq = icmp eq i64 %add, 0
  br i1 %eq, label %exit, label %loop.begin

exit:
  ret void
}

!0 = !{i32 0, i32 100}
