; RUN: opt -mtriple=armv7 -mcpu=cortex-a57 -loop-unroll -S %s -o - | FileCheck %s --check-prefix=CHECK-UNROLL-V7
; RUN: opt -mtriple=thumbv7 -mcpu=cortex-a57 -loop-unroll -S %s -o - | FileCheck %s --check-prefix=CHECK-UNROLL-V7
; RUN: opt -mtriple=thumbv8m -mcpu=cortex-m23 -loop-unroll -S %s -o - | FileCheck %s --check-prefix=CHECK-UNROLL-SMALL
; RUN: opt -mtriple=thumbv7m -mcpu=cortex-m4 -loop-unroll -S %s -o - | FileCheck %s --check-prefix=CHECK-UNROLL
; RUN: opt -mtriple=thumbv8m.main -mcpu=cortex-m33 -loop-unroll -S %s -o - | FileCheck %s --check-prefix=CHECK-UNROLL

; CHECK-LABEL: partial
define arm_aapcs_vfpcc void @partial(i32* nocapture %C, i32* nocapture readonly %A, i32* nocapture readonly %B) local_unnamed_addr #0 {
entry:
  br label %for.body

; CHECK-LABEL: for.body
for.body:

; CHECK-UNROLL-V7: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV2:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-V7: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-V7: [[IV2]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL-V7: [[CMP:%[a-z.0-9]+]] = icmp eq i32 [[IV2]], 1024
; CHECK-UNROLL-V7: br i1 [[CMP]], label [[END:%[a-z.]+]], label %for.body

; CHECK-UNROLL-SMALL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV8:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-SMALL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-SMALL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL-SMALL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL-SMALL: [[IV4:%[a-z.0-9]+]] = add nuw nsw i32 [[IV3]], 1
; CHECK-UNROLL-SMALL: [[IV5:%[a-z.0-9]+]] = add nuw nsw i32 [[IV4]], 1
; CHECK-UNROLL-SMALL: [[IV6:%[a-z.0-9]+]] = add nuw nsw i32 [[IV5]], 1
; CHECK-UNROLL-SMALL: [[IV7:%[a-z.0-9]+]] = add nuw nsw i32 [[IV6]], 1
; CHECK-UNROLL-SMALL: [[IV8]] = add nuw nsw i32 [[IV7]], 1
; CHECK-UNROLL-SMALL: [[CMP:%[a-z.0-9]+]] = icmp eq i32 [[IV8]], 1024
; CHECK-UNROLL-SMALL: br i1 [[CMP]], label [[END:%[a-z.]+]], label %for.body

; CHECK-UNROLL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV16:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL: [[IV4:%[a-z.0-9]+]] = add nuw nsw i32 [[IV3]], 1
; CHECK-UNROLL: [[IV5:%[a-z.0-9]+]] = add nuw nsw i32 [[IV4]], 1
; CHECK-UNROLL: [[IV6:%[a-z.0-9]+]] = add nuw nsw i32 [[IV5]], 1
; CHECK-UNROLL: [[IV7:%[a-z.0-9]+]] = add nuw nsw i32 [[IV6]], 1
; CHECK-UNROLL: [[IV8:%[a-z.0-9]+]] = add nuw nsw i32 [[IV7]], 1
; CHECK-UNROLL: [[IV9:%[a-z.0-9]+]] = add nuw nsw i32 [[IV8]], 1
; CHECK-UNROLL: [[IV10:%[a-z.0-9]+]] = add nuw nsw i32 [[IV9]], 1
; CHECK-UNROLL: [[IV11:%[a-z.0-9]+]] = add nuw nsw i32 [[IV10]], 1
; CHECK-UNROLL: [[IV12:%[a-z.0-9]+]] = add nuw nsw i32 [[IV11]], 1
; CHECK-UNROLL: [[IV13:%[a-z.0-9]+]] = add nuw nsw i32 [[IV12]], 1
; CHECK-UNROLL: [[IV14:%[a-z.0-9]+]] = add nuw nsw i32 [[IV13]], 1
; CHECK-UNROLL: [[IV15:%[a-z.0-9]+]] = add nuw nsw i32 [[IV14]], 1
; CHECK-UNROLL: [[IV16]] = add nuw nsw i32 [[IV15]], 1
; CHECK-UNROLL: [[CMP:%[a-z.0-9]+]] = icmp eq i32 [[IV16]], 1024
; CHECK-UNROLL: br i1 [[CMP]], label [[END:%[a-z.]+]], label %for.body

  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i32 %i.08
  %1 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i32 %i.08
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-LABEL: runtime
define arm_aapcs_vfpcc void @runtime(i32* nocapture %C, i32* nocapture readonly %A, i32* nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body

; CHECK-LABEL: for.body
for.body:
; CHECK-UNROLL-V7: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z.0-9]+]] ], [ [[IV8:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-V7: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-V7: [[IV2]] = add nuw  i32 [[IV1]], 1

; CHECK-UNROLL-SMALL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z.0-9]+]] ], [ [[IV8:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-SMALL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-SMALL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL-SMALL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL-SMALL: [[IV4:%[a-z.0-9]+]] = add nuw nsw i32 [[IV3]], 1
; CHECK-UNROLL-SMALL: [[IV5:%[a-z.0-9]+]] = add nuw nsw i32 [[IV4]], 1
; CHECK-UNROLL-SMALL: [[IV6:%[a-z.0-9]+]] = add nuw nsw i32 [[IV5]], 1
; CHECK-UNROLL-SMALL: [[IV7:%[a-z.0-9]+]] = add nuw nsw i32 [[IV6]], 1
; CHECK-UNROLL-SMALL: [[IV8]] = add nuw i32 [[IV7]], 1

; CHECK-UNROLL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z.0-9]+]] ], [ [[IV8:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL: [[IV4:%[a-z.0-9]+]] = add nuw nsw i32 [[IV3]], 1
; CHECK-UNROLL: [[IV5:%[a-z.0-9]+]] = add nuw nsw i32 [[IV4]], 1
; CHECK-UNROLL: [[IV6:%[a-z.0-9]+]] = add nuw nsw i32 [[IV5]], 1
; CHECK-UNROLL: [[IV7:%[a-z.0-9]+]] = add nuw nsw i32 [[IV6]], 1
; CHECK-UNROLL: [[IV8]] = add nuw i32 [[IV7]], 1

  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i32 %i.09
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-LABEL: nested_runtime
define arm_aapcs_vfpcc void @nested_runtime(i32* nocapture %C, i16* nocapture readonly %A, i16* nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp25 = icmp eq i32 %N, 0
  br i1 %cmp25, label %for.cond.cleanup, label %for.body4.lr.ph

for.body4.lr.ph:
  %h.026 = phi i32 [ %inc11, %for.cond.cleanup3 ], [ 0, %entry ]
  %mul = mul i32 %h.026, %N
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %inc11 = add nuw i32 %h.026, 1
  %exitcond27 = icmp eq i32 %inc11, %N
  br i1 %exitcond27, label %for.cond.cleanup, label %for.body4.lr.ph

; CHECK-LABEL: for.body4
for.body4:
; CHECK-UNROLL-V7: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z0-9.]+]] ], [ [[IV1:%[a-z.0-9]+]], %for.body4 ]
; CHECK-UNROLL-V7: [[IV1:%[a-z.0-9]+]] = add nuw  i32 [[IV0]], 1

; CHECK-UNROLL-SMALL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z0-9.]+]] ], [ [[IV4:%[a-z.0-9]+]], %for.body4 ]
; CHECK-UNROLL-SMALL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-SMALL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL-SMALL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL-SMALL: [[IV4]] = add nuw i32 [[IV3]], 1

; CHECK-UNROLL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, [[PRE:%[a-z0-9.]+]] ], [ [[IV8:%[a-z.0-9]+]], %for.body4 ]
; CHECK-UNROLL: [[IV1:%[a-z.0-9]+]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL: [[IV2:%[a-z.0-9]+]] = add nuw nsw i32 [[IV1]], 1
; CHECK-UNROLL: [[IV3:%[a-z.0-9]+]] = add nuw nsw i32 [[IV2]], 1
; CHECK-UNROLL: [[IV4:%[a-z.0-9]+]] = add nuw nsw i32 [[IV3]], 1
; CHECK-UNROLL: [[IV5:%[a-z.0-9]+]] = add nuw nsw i32 [[IV4]], 1
; CHECK-UNROLL: [[IV6:%[a-z.0-9]+]] = add nuw nsw i32 [[IV5]], 1
; CHECK-UNROLL: [[IV7:%[a-z.0-9]+]] = add nuw nsw i32 [[IV6]], 1
; CHECK-UNROLL: [[IV8:%[a-z.0-9]+]] = add nuw i32 [[IV7]], 1

  %w.024 = phi i32 [ 0, %for.body4.lr.ph ], [ %inc, %for.body4 ]
  %add = add i32 %w.024, %mul
  %arrayidx = getelementptr inbounds i16, i16* %A, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %arrayidx5 = getelementptr inbounds i16, i16* %B, i32 %w.024
  %1 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %1 to i32
  %mul7 = mul nsw i32 %conv6, %conv
  %arrayidx8 = getelementptr inbounds i32, i32* %C, i32 %w.024
  %2 = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %mul7, %2
  store i32 %add9, i32* %arrayidx8, align 4
  %inc = add nuw i32 %w.024, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup3, label %for.body4
}

; CHECK-LABEL: loop_call
define arm_aapcs_vfpcc void @loop_call(i32* nocapture %C, i32* nocapture readonly %A, i32* nocapture readonly %B) local_unnamed_addr #1 {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

; CHECK-LABEL: for.body
for.body:
; CHECK-UNROLL-V7: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV1:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-V7: [[IV1]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-V7: icmp eq i32 [[IV1]], 1024

; CHECK-UNROLL-SMALL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV1:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL-SMALL: [[IV1]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL-SMALL: icmp eq i32 [[IV1]], 1024

; CHECK-UNROLL: [[IV0:%[a-z.0-9]+]] = phi i32 [ 0, %entry ], [ [[IV1:%[a-z.0-9]+]], %for.body ]
; CHECK-UNROLL: [[IV1]] = add nuw nsw i32 [[IV0]], 1
; CHECK-UNROLL: icmp eq i32 [[IV1]], 1024

  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i32 %i.08
  %1 = load i32, i32* %arrayidx1, align 4
  %call = tail call arm_aapcs_vfpcc i32 @some_func(i32 %0, i32 %1) #3
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i32 %i.08
  store i32 %call, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare arm_aapcs_vfpcc i32 @some_func(i32, i32) local_unnamed_addr #2
