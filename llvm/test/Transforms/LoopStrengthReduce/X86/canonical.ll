; RUN: opt -mtriple=x86_64-unknown-linux-gnu -loop-reduce -lsr-insns-cost=false -S < %s | FileCheck %s
; Check LSR formula canonicalization will put loop invariant regs before
; induction variable of current loop, so exprs involving loop invariant regs
; can be promoted outside of current loop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32 %size, i32 %nsteps, i8* nocapture %maxarray, i8* nocapture readnone %buffer, i32 %init) local_unnamed_addr #0 {
entry:
  %cmp25 = icmp sgt i32 %nsteps, 0
  br i1 %cmp25, label %for.cond1.preheader.lr.ph, label %for.end12

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp223 = icmp sgt i32 %size, 1
  %t0 = sext i32 %init to i64
  %wide.trip.count = zext i32 %size to i64
  %wide.trip.count31 = zext i32 %nsteps to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc10, %for.cond1.preheader.lr.ph
  %indvars.iv28 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next29, %for.inc10 ]
  br i1 %cmp223, label %for.body3.lr.ph, label %for.inc10

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %t1 = add nsw i64 %indvars.iv28, %t0
  %t2 = trunc i64 %indvars.iv28 to i8
  br label %for.body3

; Make sure loop invariant items are grouped together so that load address can
; be represented in one getelementptr.
; CHECK-LABEL: for.body3:
; CHECK-NEXT: [[LSR:%[^,]+]] = phi i64 [ 1, %for.body3.lr.ph ], [ {{.*}}, %for.body3 ]
; CHECK-NOT: = phi i64
; CHECK-NEXT: [[LOADADDR:%[^,]+]] = getelementptr i8, i8* {{.*}}, i64 [[LSR]]
; CHECK-NEXT: = load i8, i8* [[LOADADDR]], align 1
; CHECK: br i1 %exitcond, label %for.inc10.loopexit, label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %t5 = trunc i64 %indvars.iv to i8
  %t3 = add nsw i64 %t1, %indvars.iv
  %arrayidx = getelementptr inbounds i8, i8* %maxarray, i64 %t3
  %t4 = load i8, i8* %arrayidx, align 1
  %add5 = add i8 %t4, %t5
  %add6 = add i8 %add5, %t2
  %arrayidx9 = getelementptr inbounds i8, i8* %maxarray, i64 %indvars.iv
  store i8 %add6, i8* %arrayidx9, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.inc10.loopexit, label %for.body3

for.inc10.loopexit:                               ; preds = %for.body3
  br label %for.inc10

for.inc10:                                        ; preds = %for.inc10.loopexit, %for.cond1.preheader
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond32 = icmp eq i64 %indvars.iv.next29, %wide.trip.count31
  br i1 %exitcond32, label %for.end12.loopexit, label %for.cond1.preheader

for.end12.loopexit:                               ; preds = %for.inc10
  br label %for.end12

for.end12:                                        ; preds = %for.end12.loopexit, %entry
  ret void
}
