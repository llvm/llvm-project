; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; Inner loop has a store to invariant address, but LAA does not need to identify
; the store to invariant address, since it is a single store.
;
;  for(; i < itr; i++) {
;    for(; j < itr; j++) {
;      var1[j] = ++var2[i] + var1[j];
;    }
;  }

; CHECK: Non vectorizable stores to invariant address were not found in loop.

define void @foo(ptr nocapture %var1, ptr nocapture %var2, i32 %itr) #0 {
entry:
  %cmp20 = icmp sgt i32 %itr, 0
  br i1 %cmp20, label %for.cond1.preheader, label %for.end11

for.cond1.preheader:                              ; preds = %entry, %for.inc9
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %for.inc9 ], [ 0, %entry ]
  %j.022 = phi i32 [ %j.1.lcssa, %for.inc9 ], [ 0, %entry ]
  %cmp218 = icmp slt i32 %j.022, %itr
  br i1 %cmp218, label %for.body3.lr.ph, label %for.inc9

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %arrayidx = getelementptr inbounds i32, ptr %var2, i64 %indvars.iv23
  %0 = sext i32 %j.022 to i64
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ %0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %1 = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds i32, ptr %var1, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx5, align 4
  %add = add nsw i32 %inc, %2
  store i32 %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.inc9, label %for.body3

for.inc9:                                         ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.022, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv.next24 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %itr
  br i1 %exitcond26, label %for.end11, label %for.cond1.preheader

for.end11:                                        ; preds = %for.inc9, %entry
  ret void
}
