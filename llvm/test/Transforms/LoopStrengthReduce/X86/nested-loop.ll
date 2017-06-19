; RUN: opt -loop-reduce -S < %s | FileCheck %s
; Check when we use an outerloop induction variable inside of an innerloop
; induction value expr, LSR can still choose to use single induction variable
; for the innerloop and share it in multiple induction value exprs.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %size, i32 %nsteps, i32 %hsize, i32* %lined, i8* %maxarray) {
entry:
  %cmp215 = icmp sgt i32 %size, 1
  %t0 = zext i32 %size to i64
  %t1 = sext i32 %nsteps to i64
  %sub2 = sub i64 %t0, 2
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %for.inc ], [ 0, %entry ]
  %t2 = mul nsw i64 %indvars.iv2, %t0
  br i1 %cmp215, label %for.body2.preheader, label %for.inc

for.body2.preheader:                              ; preds = %for.body
  br label %for.body2

; Check LSR only generates one induction variable for for.body2 and the induction
; variable will be shared by multiple array accesses.
; CHECK: for.body2:
; CHECK-NEXT: [[LSR:%[^,]+]] = phi i64 [ %lsr.iv.next, %for.body2 ], [ 0, %for.body2.preheader ]
; CHECK-NOT:  = phi i64 [ {{.*}}, %for.body2 ], [ {{.*}}, %for.body2.preheader ]
; CHECK:      [[SCEVGEP1:%[^,]+]] = getelementptr i8, i8* %maxarray, i64 [[LSR]]
; CHECK:      [[SCEVGEP2:%[^,]+]] = getelementptr i8, i8* [[SCEVGEP1]], i64 1
; CHECK:      {{.*}} = load i8, i8* [[SCEVGEP2]], align 1
; CHECK:      [[SCEVGEP3:%[^,]+]] = getelementptr i8, i8* {{.*}}, i64 [[LSR]]
; CHECK:      {{.*}} = load i8, i8* [[SCEVGEP3]], align 1
; CHECK:      [[SCEVGEP4:%[^,]+]] = getelementptr i8, i8* {{.*}}, i64 [[LSR]]
; CHECK:      store i8 {{.*}}, i8* [[SCEVGEP4]], align 1
; CHECK:      br i1 %exitcond, label %for.body2, label %for.inc.loopexit

for.body2:                                        ; preds = %for.body2.preheader, %for.body2
  %indvars.iv = phi i64 [ 1, %for.body2.preheader ], [ %indvars.iv.next, %for.body2 ]
  %arrayidx1 = getelementptr inbounds i8, i8* %maxarray, i64 %indvars.iv
  %v1 = load i8, i8* %arrayidx1, align 1
  %idx2 = add nsw i64 %indvars.iv, %sub2
  %arrayidx2 = getelementptr inbounds i8, i8* %maxarray, i64 %idx2
  %v2 = load i8, i8* %arrayidx2, align 1
  %tmpv = xor i8 %v1, %v2
  %t4 = add nsw i64 %t2, %indvars.iv
  %add.ptr = getelementptr inbounds i8, i8* %maxarray, i64 %t4
  store i8 %tmpv, i8* %add.ptr, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %wide.trip.count = zext i32 %size to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body2, label %for.inc.loopexit

for.inc.loopexit:                                 ; preds = %for.body2
  br label %for.inc

for.inc:                                          ; preds = %for.inc.loopexit, %for.body
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  %cmp = icmp slt i64 %indvars.iv.next3, %t1
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.inc
  ret void
}
