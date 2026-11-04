; RUN: opt -disable-output -passes='print<access-info>' %s 2>&1 | FileCheck %s --check-prefixes=CHECK,FULLDEPTH
; RUN: opt -disable-output -passes='print<access-info>' -max-forked-scev-depth=2 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DEPTH2

define void @laa_common_output(ptr %Base, ptr %Dest, ptr %Preds) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %i.014 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %add = add nuw nsw i32 %i.014, 1
  %1 = trunc i64 %indvars.iv to i32
  %offset.0 = select i1 %cmp1.not, i32 %add, i32 %1
  %idxprom213 = zext i32 %offset.0 to i64
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %idxprom213
  %2 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %2, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @laa_divergent_output(ptr %Base, ptr %Dest, ptr %Preds, i64 %extra_offset) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp.not = icmp eq i32 %0, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %sel = select i1 %cmp.not, i64 %indvars.iv.next, i64 %indvars.iv
  %offset = add nuw nsw i64 %sel, %extra_offset
  %arrayidx3 = getelementptr inbounds float, ptr %Base, i64 %offset
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %Dest, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
