; RUN: opt -loop-vectorize < %s -S -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; CHECK-LABEL: check4
; CHECK: call i32 @llvm.experimental.vector.reduce.add.v4i32
define i32 @check4(i8* noalias nocapture readonly %A, i8* noalias nocapture readonly %B, i32 %n) #0 {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ undef, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.010 = phi i32 [ %add, %for.body ], [ undef, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i32 %i.011
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.011
  %1 = load i8, i8* %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.010
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: check16
; CHECK: call i32 @llvm.experimental.vector.reduce.add.v16i32
define i32 @check16(i8* noalias nocapture readonly %A, i8* noalias nocapture readonly %B, i32 %n) #0 {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ undef, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %res.010 = phi i32 [ %add, %for.body ], [ undef, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i32 %i.011
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8, i8* %B, i32 %i.011
  %1 = load i8, i8* %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %res.010
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}

attributes #0 = { "target-features"="+mve" }
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.vectorize.width", i32 16}
