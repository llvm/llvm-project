; RUN: opt -mtriple=thumbv7em -arm-parallel-dsp -dce -S %s -o - | FileCheck %s

; CHECK-LABEL: full_unroll
; CHECK: [[IV:%[^ ]+]] = phi i32
; CHECK: [[AI:%[^ ]+]] = getelementptr inbounds i32, i32* %a, i32 [[IV]]
; CHECK: [[BI:%[^ ]+]] = getelementptr inbounds i16*, i16** %b, i32 [[IV]]
; CHECK: [[BIJ:%[^ ]+]] = load i16*, i16** %arrayidx5, align 4
; CHECK: [[CI:%[^ ]+]] = getelementptr inbounds i16*, i16** %c, i32 [[IV]]
; CHECK: [[CIJ:%[^ ]+]] = load i16*, i16** [[CI]], align 4
; CHECK: [[BIJ_CAST:%[^ ]+]] = bitcast i16* [[BIJ]] to i32*
; CHECK: [[BIJ_LD:%[^ ]+]] = load i32, i32* [[BIJ_CAST]], align 2
; CHECK: [[CIJ_CAST:%[^ ]+]] = bitcast i16* [[CIJ]] to i32*
; CHECK: [[CIJ_LD:%[^ ]+]] = load i32, i32* [[CIJ_CAST]], align 2
; CHECK: [[SMLAD0:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[CIJ_LD]], i32 [[BIJ_LD]], i32 0)
; CHECK: [[BIJ_2:%[^ ]+]] = getelementptr inbounds i16, i16* [[BIJ]], i32 2
; CHECK: [[BIJ_2_CAST:%[^ ]+]] = bitcast i16* [[BIJ_2]] to i32*
; CHECK: [[BIJ_2_LD:%[^ ]+]] = load i32, i32* [[BIJ_2_CAST]], align 2
; CHECK: [[CIJ_2:%[^ ]+]] = getelementptr inbounds i16, i16* [[CIJ]], i32 2
; CHECK: [[CIJ_2_CAST:%[^ ]+]] = bitcast i16* [[CIJ_2]] to i32*
; CHECK: [[CIJ_2_LD:%[^ ]+]] = load i32, i32* [[CIJ_2_CAST]], align 2
; CHECK: [[SMLAD1:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[CIJ_2_LD]], i32 [[BIJ_2_LD]], i32 [[SMLAD0]])
; CHECK: store i32 [[SMLAD1]], i32* %arrayidx, align 4

define void @full_unroll(i32* noalias nocapture %a, i16** noalias nocapture readonly %b, i16** noalias nocapture readonly %c, i32 %N) {
entry:
  %cmp29 = icmp eq i32 %N, 0
  br i1 %cmp29, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.030 = phi i32 [ %inc12, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %i.030
  %arrayidx5 = getelementptr inbounds i16*, i16** %b, i32 %i.030
  %0 = load i16*, i16** %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i16*, i16** %c, i32 %i.030
  %1 = load i16*, i16** %arrayidx7, align 4
  %2 = load i16, i16* %0, align 2
  %conv = sext i16 %2 to i32
  %3 = load i16, i16* %1, align 2
  %conv9 = sext i16 %3 to i32
  %mul = mul nsw i32 %conv9, %conv
  %arrayidx6.1 = getelementptr inbounds i16, i16* %0, i32 1
  %4 = load i16, i16* %arrayidx6.1, align 2
  %conv.1 = sext i16 %4 to i32
  %arrayidx8.1 = getelementptr inbounds i16, i16* %1, i32 1
  %5 = load i16, i16* %arrayidx8.1, align 2
  %conv9.1 = sext i16 %5 to i32
  %mul.1 = mul nsw i32 %conv9.1, %conv.1
  %add.1 = add nsw i32 %mul.1, %mul
  %arrayidx6.2 = getelementptr inbounds i16, i16* %0, i32 2
  %6 = load i16, i16* %arrayidx6.2, align 2
  %conv.2 = sext i16 %6 to i32
  %arrayidx8.2 = getelementptr inbounds i16, i16* %1, i32 2
  %7 = load i16, i16* %arrayidx8.2, align 2
  %conv9.2 = sext i16 %7 to i32
  %mul.2 = mul nsw i32 %conv9.2, %conv.2
  %add.2 = add nsw i32 %mul.2, %add.1
  %arrayidx6.3 = getelementptr inbounds i16, i16* %0, i32 3
  %8 = load i16, i16* %arrayidx6.3, align 2
  %conv.3 = sext i16 %8 to i32
  %arrayidx8.3 = getelementptr inbounds i16, i16* %1, i32 3
  %9 = load i16, i16* %arrayidx8.3, align 2
  %conv9.3 = sext i16 %9 to i32
  %mul.3 = mul nsw i32 %conv9.3, %conv.3
  %add.3 = add nsw i32 %mul.3, %add.2
  store i32 %add.3, i32* %arrayidx, align 4
  %inc12 = add nuw i32 %i.030, 1
  %exitcond = icmp eq i32 %inc12, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: full_unroll_sub
; CHEC: [[IV:%[^ ]+]] = phi i32
; CHECK: [[AI:%[^ ]+]] = getelementptr inbounds i32, i32* %a, i32 [[IV]]
; CHECK: [[BI:%[^ ]+]] = getelementptr inbounds i16*, i16** %b, i32 [[IV]]
; CHECK: [[BIJ:%[^ ]+]] = load i16*, i16** [[BI]], align 4
; CHECK: [[CI:%[^ ]+]] = getelementptr inbounds i16*, i16** %c, i32 [[IV]]
; CHECK: [[CIJ:%[^ ]+]] = load i16*, i16** [[CI]], align 4
; CHECK: [[BIJ_LD:%[^ ]+]] = load i16, i16* [[BIJ]], align 2
; CHECK: [[BIJ_LD_SXT:%[^ ]+]] = sext i16 [[BIJ_LD]] to i32
; CHECK: [[CIJ_LD:%[^ ]+]] = load i16, i16* [[CIJ]], align 2
; CHECK: [[CIJ_LD_SXT:%[^ ]+]] = sext i16 [[CIJ_LD]] to i32
; CHECK: [[SUB:%[^ ]+]] = sub nsw i32 [[CIJ_LD_SXT]], [[BIJ_LD_SXT]]
; CHECK: [[BIJ_1:%[^ ]+]] = getelementptr inbounds i16, i16* [[BIJ]], i32 1
; CHECK: [[BIJ_1_LD:%[^ ]+]] = load i16, i16* [[BIJ_1]], align 2
; CHECK: [[BIJ_1_LD_SXT:%[^ ]+]] = sext i16 [[BIJ_1_LD]] to i32
; CHECK: [[CIJ_1:%[^ ]+]] = getelementptr inbounds i16, i16* [[CIJ]], i32 1
; CHECK: [[CIJ_1_LD:%[^ ]+]] = load i16, i16* [[CIJ_1]], align 2
; CHECK: [[CIJ_1_LD_SXT:%[^ ]+]] = sext i16 [[CIJ_1_LD]] to i32
; CHECK: [[MUL:%[^ ]+]] = mul nsw i32 [[CIJ_1_LD_SXT]], [[BIJ_1_LD_SXT]]
; CHECK: [[ACC:%[^ ]+]] = add nsw i32 [[MUL]], [[SUB]]
; CHECK: [[BIJ_2:%[^ ]+]] = getelementptr inbounds i16, i16* [[BIJ]], i32 2
; CHECK: [[BIJ_2_CAST:%[^ ]+]] = bitcast i16* [[BIJ_2]] to i32*
; CHECK: [[BIJ_2_LD:%[^ ]+]] = load i32, i32* [[BIJ_2_CAST]], align 2
; CHECK: [[CIJ_2:%[^ ]+]] = getelementptr inbounds i16, i16* [[CIJ]], i32 2
; CHECK: [[CIJ_2_CAST:%[^ ]+]] = bitcast i16* [[CIJ_2]] to i32*
; CHECK: [[CIJ_2_LD:%[^ ]+]] = load i32, i32* [[CIJ_2_CAST]], align 2
; CHECK: [[SMLAD0:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[CIJ_2_LD]], i32 [[BIJ_2_LD]], i32 [[ACC]])
; CHECK: store i32 [[SMLAD0]], i32* %arrayidx, align 4

define void @full_unroll_sub(i32* noalias nocapture %a, i16** noalias nocapture readonly %b, i16** noalias nocapture readonly %c, i32 %N) {
entry:
  %cmp29 = icmp eq i32 %N, 0
  br i1 %cmp29, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.030 = phi i32 [ %inc12, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %i.030
  %arrayidx5 = getelementptr inbounds i16*, i16** %b, i32 %i.030
  %0 = load i16*, i16** %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i16*, i16** %c, i32 %i.030
  %1 = load i16*, i16** %arrayidx7, align 4
  %2 = load i16, i16* %0, align 2
  %conv = sext i16 %2 to i32
  %3 = load i16, i16* %1, align 2
  %conv9 = sext i16 %3 to i32
  %sub = sub nsw i32 %conv9, %conv
  %arrayidx6.1 = getelementptr inbounds i16, i16* %0, i32 1
  %4 = load i16, i16* %arrayidx6.1, align 2
  %conv.1 = sext i16 %4 to i32
  %arrayidx8.1 = getelementptr inbounds i16, i16* %1, i32 1
  %5 = load i16, i16* %arrayidx8.1, align 2
  %conv9.1 = sext i16 %5 to i32
  %mul.1 = mul nsw i32 %conv9.1, %conv.1
  %add.1 = add nsw i32 %mul.1, %sub
  %arrayidx6.2 = getelementptr inbounds i16, i16* %0, i32 2
  %6 = load i16, i16* %arrayidx6.2, align 2
  %conv.2 = sext i16 %6 to i32
  %arrayidx8.2 = getelementptr inbounds i16, i16* %1, i32 2
  %7 = load i16, i16* %arrayidx8.2, align 2
  %conv9.2 = sext i16 %7 to i32
  %mul.2 = mul nsw i32 %conv9.2, %conv.2
  %add.2 = add nsw i32 %mul.2, %add.1
  %arrayidx6.3 = getelementptr inbounds i16, i16* %0, i32 3
  %8 = load i16, i16* %arrayidx6.3, align 2
  %conv.3 = sext i16 %8 to i32
  %arrayidx8.3 = getelementptr inbounds i16, i16* %1, i32 3
  %9 = load i16, i16* %arrayidx8.3, align 2
  %conv9.3 = sext i16 %9 to i32
  %mul.3 = mul nsw i32 %conv9.3, %conv.3
  %add.3 = add nsw i32 %mul.3, %add.2
  store i32 %add.3, i32* %arrayidx, align 4
  %inc12 = add nuw i32 %i.030, 1
  %exitcond = icmp eq i32 %inc12, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
