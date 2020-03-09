; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-count=4 -unroll-remainder -instcombine | FileCheck %s

; CHECK-LABEL: unroll
define i32 @unroll(i32* nocapture readonly %a, i32* nocapture readonly %b, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp9 = icmp eq i32 %N, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.cond.cleanup:
  %c.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %c.0.lcssa

; CHECK-LABEL: for.body.lr.ph
; CHECK: [[COUNT:%[a-z.0-9]+]] = add nsw i64 %wide.trip.count, -1
; CHECK: %xtraiter = and i64 %wide.trip.count, 3
; CHECK: [[CMP:%[a-z.0-9]+]] = icmp ult i64 [[COUNT]], 3
; CHECK: br i1 [[CMP]], label %[[CLEANUP:.*]], label %for.body.lr.ph.new

; CHECK-LABEL: for.body.lr.ph.new:
; CHECK: %unroll_iter = and i64 %wide.trip.count, 4294967292
; CHECK: br label %for.body

; CHECK: [[CLEANUP]]:
; CHECK: [[MOD:%[a-z.0-9]+]] = icmp eq i64 %xtraiter, 0
; CHECK: br i1 [[MOD]], label %[[EXIT:.*]], label %[[EPIL_PEEL0_PRE:.*]]

; CHECK: [[EPIL_PEEL0_PRE]]:
; CHECK: br label %[[EPIL_PEEL0:.*]]

; CHECK: [[EPIL_PEEL0]]:
; CHECK: [[PEEL_CMP0:%[a-z.0-9]+]] = icmp eq i64 %xtraiter, 1
; CHECK: br i1 [[PEEL_CMP0]], label %[[EPIL_EXIT:.*]], label %[[EPIL_PEEL1:.*]]

; CHECK: [[EPIL_EXIT]]:
; CHECK: br label %[[EXIT]]

; CHECK: [[EXIT]]:
; CHECK: ret i32

; CHECK-LABEL: for.body:
; CHECK: [[INDVAR0:%[a-z.0-9]+]] = phi i64 [ 0, %for.body.lr.ph
; CHECK: [[ITER:%[a-z.0-9]+]] = phi i64 [ %unroll_iter
; CHECK: or i64 [[INDVAR0]], 1
; CHECK: or i64 [[INDVAR0]], 2
; CHECK: or i64 [[INDVAR0]], 3
; CHECK: add nuw nsw i64 [[INDVAR0]], 4
; CHECK: [[SUB:%[a-z.0-9]+]] = add i64 [[ITER]], -4
; CHECK: [[ITER_CMP:%[a-z.0-9]+]] = icmp eq i64 [[SUB]], 0
; CHECK: br i1 [[ITER_CMP]], label %[[LOOP_EXIT:.*]], label %for.body

; CHECK: [[EPIL_PEEL1]]:
; CHECK: [[PEEL_CMP1:%[a-z.0-9]+]] = icmp eq i64 %xtraiter, 2
; CHECK: br i1 [[PEEL_CMP1]], label %[[EPIL_EXIT]], label %[[EPIL_PEEL2:.*]]

; CHECK: [[EPIL_PEEL2]]:
; CHECK: br label %[[EXIT]]

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %c.010 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %c.010
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
