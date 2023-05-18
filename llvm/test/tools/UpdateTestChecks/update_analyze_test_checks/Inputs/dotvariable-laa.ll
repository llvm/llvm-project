; RUN: opt -passes='print<access-info>' < %s -disable-output 2>&1 | FileCheck %s

define dso_local void @dotvariable_laa(ptr nocapture readonly nonnull %Base1, ptr nocapture readonly %Base2, ptr nocapture %Dest, ptr nocapture readonly %Preds) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Preds, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1.not = icmp eq i32 %0, 0
  %spec.select = select i1 %cmp1.not, ptr %Base2, ptr %Base1
  %.sink.in = getelementptr inbounds double, ptr %spec.select, i64 %indvars.iv
  %.sink = load double, ptr %.sink.in, align 8
  %1 = getelementptr inbounds double, ptr %Dest, i64 %indvars.iv
  store double %.sink, ptr %1, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
