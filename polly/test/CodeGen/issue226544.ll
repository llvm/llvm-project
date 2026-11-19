; RUN: opt %loadNPMPolly -passes=polly -pass-remarks-missed=polly-codegen -S %s 2>&1 | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/226544
;
; When Polly tiles loop schedules involving large parameters and constants,
; ISL AST expressions can exceed signed 64-bit integer arithmetic capacity.
; Verify that Polly detects the overflow, bails out, and executes the original un-optimized code.
;
; CHECK: remark: {{.*}} Bailing out: AST expression exceeds 64-bit integer arithmetic
; CHECK-LABEL: polly.split_new_and_old
; CHECK:       br i1 false, label %[[START:polly\.start[0-9]*]], label %for.body4.us.us.us.pre_entry_bb
; CHECK:       [[START]]:
; CHECK-NEXT:  unreachable

@arr_52 = external local_unnamed_addr global [10 x [10 x [10 x i64]]], align 8
@arr_118 = external local_unnamed_addr global [11 x [11 x [11 x i16]]], align 2

define void @test(i64 noundef %var_5, i8 noundef signext %var_9) local_unnamed_addr {
entry:
  %sub = add i64 %var_5, -600000000000000000
  %cmp242 = icmp ugt i64 %sub, 3
  %tobool.not = icmp eq i8 %var_9, 0
  %conv24 = select i1 %tobool.not, i64 0, i64 4
  %conv = sext i8 %var_9 to i64
  %sub11 = add nsw i64 %conv, -5
  br i1 %cmp242, label %entry.split.us, label %for.cond.cleanup

entry.split.us:
  %cmp1240 = icmp sgt i8 %var_9, 5
  br i1 %cmp1240, label %for.cond1.preheader.us.us, label %for.cond1.preheader.us

for.cond1.preheader.us.us:
  %i_25.044.us.us = phi i64 [ %add25.us.us, %for.cond1.for.cond.cleanup3_crit_edge.split.us.us.us ], [ 0, %entry.split.us ]
  %arrayidx16.us.us = getelementptr inbounds nuw [242 x i8], ptr @arr_118, i64 %i_25.044.us.us
  br label %for.body4.us.us.us

for.body4.us.us.us:
  %i_26.043.us.us.us = phi i64 [ 3, %for.cond1.preheader.us.us ], [ %add20.us.us.us, %for.cond10.for.cond.cleanup14_crit_edge.us.us.us ]
  %arrayidx.us.us.us = getelementptr inbounds nuw [800 x i8], ptr @arr_52, i64 %i_26.043.us.us.us
  %arrayidx5.us.us.us = getelementptr inbounds nuw [80 x i8], ptr %arrayidx.us.us.us, i64 %i_26.043.us.us.us
  %arrayidx6.us.us.us = getelementptr inbounds nuw [8 x i8], ptr %arrayidx5.us.us.us, i64 %i_26.043.us.us.us
  %0 = load i64, ptr %arrayidx6.us.us.us, align 8
  %arrayidx8.us.us.us = getelementptr inbounds nuw [80 x i8], ptr %arrayidx.us.us.us, i64 %i_25.044.us.us
  %arrayidx9.us.us.us = getelementptr inbounds nuw [8 x i8], ptr %arrayidx8.us.us.us, i64 %i_25.044.us.us
  store i64 %0, ptr %arrayidx9.us.us.us, align 8
  %arrayidx17.us.us.us = getelementptr inbounds nuw [22 x i8], ptr %arrayidx16.us.us, i64 %i_26.043.us.us.us
  br label %for.body15.us.us.us

for.body15.us.us.us:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body15.us.us.us ], [ 0, %for.body4.us.us.us ]
  %arrayidx18.us.us.us = getelementptr inbounds nuw [2 x i8], ptr %arrayidx17.us.us.us, i64 %indvars.iv
  store i16 1, ptr %arrayidx18.us.us.us, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp12.us.us.us = icmp slt i64 %indvars.iv.next, %sub11
  br i1 %cmp12.us.us.us, label %for.body15.us.us.us, label %for.cond10.for.cond.cleanup14_crit_edge.us.us.us

for.cond10.for.cond.cleanup14_crit_edge.us.us.us:
  %add20.us.us.us = add i64 %i_26.043.us.us.us, 3
  %cmp2.us.us.us = icmp ult i64 %add20.us.us.us, %sub
  br i1 %cmp2.us.us.us, label %for.body4.us.us.us, label %for.cond1.for.cond.cleanup3_crit_edge.split.us.us.us

for.cond1.for.cond.cleanup3_crit_edge.split.us.us.us:
  %add25.us.us = add nuw nsw i64 %i_25.044.us.us, %conv24
  %cmp.us.us = icmp samesign ult i64 %add25.us.us, 11
  br i1 %cmp.us.us, label %for.cond1.preheader.us.us, label %for.cond.cleanup

for.cond1.preheader.us:
  %i_25.044.us = phi i64 [ %add25.us, %for.cond1.for.cond.cleanup3_crit_edge.split.us56 ], [ 0, %entry.split.us ]
  br label %for.body4.us47

for.body4.us47:
  %i_26.043.us48 = phi i64 [ 3, %for.cond1.preheader.us ], [ %add20.us54, %for.body4.us47 ]
  %arrayidx.us49 = getelementptr inbounds nuw [800 x i8], ptr @arr_52, i64 %i_26.043.us48
  %arrayidx5.us50 = getelementptr inbounds nuw [80 x i8], ptr %arrayidx.us49, i64 %i_26.043.us48
  %arrayidx6.us51 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx5.us50, i64 %i_26.043.us48
  %1 = load i64, ptr %arrayidx6.us51, align 8
  %arrayidx8.us52 = getelementptr inbounds nuw [80 x i8], ptr %arrayidx.us49, i64 %i_25.044.us
  %arrayidx9.us53 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx8.us52, i64 %i_25.044.us
  store i64 %1, ptr %arrayidx9.us53, align 8
  %add20.us54 = add i64 %i_26.043.us48, 3
  %cmp2.us55 = icmp ult i64 %add20.us54, %sub
  br i1 %cmp2.us55, label %for.body4.us47, label %for.cond1.for.cond.cleanup3_crit_edge.split.us56

for.cond1.for.cond.cleanup3_crit_edge.split.us56:
  %add25.us = add nuw nsw i64 %i_25.044.us, %conv24
  %cmp.us = icmp samesign ult i64 %add25.us, 11
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
