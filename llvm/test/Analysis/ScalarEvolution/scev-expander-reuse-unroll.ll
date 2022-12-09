; RUN: opt < %s -passes=loop-unroll -unroll-runtime -unroll-count=2 -verify-scev-maps -S | FileCheck %s

; Check SCEV expansion uses existing value when unrolling an inner loop with runtime trip count in a loop nest.
; The outer loop gets unrolled twice, so we see 2 selects in the outer loop blocks.
; CHECK-LABEL: @foo(
; CHECK-LABEL: for.body.loopexit:
; CHECK: select
; CHECK-LABEL: for.body:
; CHECK: select
; CHECK-NOT: select
; CHECK: ret

define void @foo(i32 %xfL, i32 %scaleL) local_unnamed_addr {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body5, %for.body, %entry
  %xfL.addr.033 = phi i32 [ %xfL, %entry ], [ %add, %for.body5 ]
  %add = add nsw i32 %xfL.addr.033, %scaleL
  %shr = ashr i32 %add, 16
  %cmp.i = icmp slt i32 10, %shr
  %.sroa.speculated = select i1 %cmp.i, i32 0, i32 %shr
  %cmp425 = icmp slt i32 0, %.sroa.speculated
  br i1 %cmp425, label %for.body5.preheader, label %for.end

for.body5.preheader:                              ; preds = %for.body
  %tmp0 = sext i32 %.sroa.speculated to i64
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %for.body5.preheader
  %indvars.iv = phi i64 [ 0, %for.body5.preheader ], [ %indvars.iv.next, %for.body5 ]
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %cmp4 = icmp slt i64 %indvars.iv.next, %tmp0
  br i1 %cmp4, label %for.body5, label %for.body

for.end:
  ret void
}

