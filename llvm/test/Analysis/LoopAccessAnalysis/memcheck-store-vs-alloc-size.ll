; RUN: opt -analyze --loop-accesses %s -enable-new-pm=0 | FileCheck %s
; RUN: opt -passes=print-access-info %s -disable-output 2>&1 | FileCheck %s

; This test defends against accidentally using alloc size instead of store size when performing run-time
; boundary check of memory accesses. The IR in this file is based on
; llvm/test/Analysis/LoopAccessAnalysis/memcheck-off-by-one-error.ll.
; Here, we use i19 instead of i64 because it has a different alloc size to its store size.

;CHECK: function 'fastCopy':
;CHECK: (Low: %op High: (27 + %op))
;CHECK: (Low: %src High: (27 + %src))

define void @fastCopy(i8* nocapture readonly %src, i8* nocapture %op) {
entry:
  br label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %len.addr.07 = phi i32 [ %sub, %while.body ], [ 32, %while.body.preheader ]
  %op.addr.06 = phi i8* [ %add.ptr1, %while.body ], [ %op, %while.body.preheader ]
  %src.addr.05 = phi i8* [ %add.ptr, %while.body ], [ %src, %while.body.preheader ]
  %0 = bitcast i8* %src.addr.05 to i19*
  %1 = load i19, i19* %0, align 8
  %2 = bitcast i8* %op.addr.06 to i19*
  store i19 %1, i19* %2, align 8
  %add.ptr = getelementptr inbounds i8, i8* %src.addr.05, i19 8
  %add.ptr1 = getelementptr inbounds i8, i8* %op.addr.06, i19 8
  %sub = add nsw i32 %len.addr.07, -8
  %cmp = icmp sgt i32 %len.addr.07, 8
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}
