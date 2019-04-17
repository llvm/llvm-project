; RUN: opt -loop-reduce -S < %s | FileCheck %s
; We find it is very bad to allow LSR formula containing SCEVAddRecExpr Reg
; from siblings of current loop. When one loop is LSR optimized, it can
; insert lsr.iv for other sibling loops, which sometimes leads to many extra
; lsr.iv inserted for loops.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@cond = common local_unnamed_addr global i64 0, align 8

; Check there is no extra lsr.iv generated in foo.
; CHECK-LABEL: @foo(
; CHECK-NOT: lsr.iv{{[0-9]+}} =
;
define void @foo(i64 %N) local_unnamed_addr {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %do.body ]
  tail call void @goo(i64 %i.0, i64 %i.0)
  %inc = add nuw nsw i64 %i.0, 1
  %t0 = load i64, i64* @cond, align 8
  %tobool = icmp eq i64 %t0, 0
  br i1 %tobool, label %do.body2.preheader, label %do.body

do.body2.preheader:                               ; preds = %do.body
  br label %do.body2

do.body2:                                         ; preds = %do.body2.preheader, %do.body2
  %i.1 = phi i64 [ %inc3, %do.body2 ], [ 0, %do.body2.preheader ]
  %j.1 = phi i64 [ %inc4, %do.body2 ], [ %inc, %do.body2.preheader ]
  tail call void @goo(i64 %i.1, i64 %j.1)
  %inc3 = add nuw nsw i64 %i.1, 1
  %inc4 = add nsw i64 %j.1, 1
  %t1 = load i64, i64* @cond, align 8
  %tobool6 = icmp eq i64 %t1, 0
  br i1 %tobool6, label %do.body8.preheader, label %do.body2

do.body8.preheader:                               ; preds = %do.body2
  br label %do.body8

do.body8:                                         ; preds = %do.body8.preheader, %do.body8
  %i.2 = phi i64 [ %inc9, %do.body8 ], [ 0, %do.body8.preheader ]
  %j.2 = phi i64 [ %inc10, %do.body8 ], [ %inc4, %do.body8.preheader ]
  tail call void @goo(i64 %i.2, i64 %j.2)
  %inc9 = add nuw nsw i64 %i.2, 1
  %inc10 = add nsw i64 %j.2, 1
  %t2 = load i64, i64* @cond, align 8
  %tobool12 = icmp eq i64 %t2, 0
  br i1 %tobool12, label %do.body14.preheader, label %do.body8

do.body14.preheader:                              ; preds = %do.body8
  br label %do.body14

do.body14:                                        ; preds = %do.body14.preheader, %do.body14
  %i.3 = phi i64 [ %inc15, %do.body14 ], [ 0, %do.body14.preheader ]
  %j.3 = phi i64 [ %inc16, %do.body14 ], [ %inc10, %do.body14.preheader ]
  tail call void @goo(i64 %i.3, i64 %j.3)
  %inc15 = add nuw nsw i64 %i.3, 1
  %inc16 = add nsw i64 %j.3, 1
  %t3 = load i64, i64* @cond, align 8
  %tobool18 = icmp eq i64 %t3, 0
  br i1 %tobool18, label %do.body20.preheader, label %do.body14

do.body20.preheader:                              ; preds = %do.body14
  br label %do.body20

do.body20:                                        ; preds = %do.body20.preheader, %do.body20
  %i.4 = phi i64 [ %inc21, %do.body20 ], [ 0, %do.body20.preheader ]
  %j.4 = phi i64 [ %inc22, %do.body20 ], [ %inc16, %do.body20.preheader ]
  tail call void @goo(i64 %i.4, i64 %j.4)
  %inc21 = add nuw nsw i64 %i.4, 1
  %inc22 = add nsw i64 %j.4, 1
  %t4 = load i64, i64* @cond, align 8
  %tobool24 = icmp eq i64 %t4, 0
  br i1 %tobool24, label %do.body26.preheader, label %do.body20

do.body26.preheader:                              ; preds = %do.body20
  br label %do.body26

do.body26:                                        ; preds = %do.body26.preheader, %do.body26
  %i.5 = phi i64 [ %inc27, %do.body26 ], [ 0, %do.body26.preheader ]
  %j.5 = phi i64 [ %inc28, %do.body26 ], [ %inc22, %do.body26.preheader ]
  tail call void @goo(i64 %i.5, i64 %j.5)
  %inc27 = add nuw nsw i64 %i.5, 1
  %inc28 = add nsw i64 %j.5, 1
  %t5 = load i64, i64* @cond, align 8
  %tobool30 = icmp eq i64 %t5, 0
  br i1 %tobool30, label %do.end31, label %do.body26

do.end31:                                         ; preds = %do.body26
  ret void
}

declare void @goo(i64, i64) local_unnamed_addr

