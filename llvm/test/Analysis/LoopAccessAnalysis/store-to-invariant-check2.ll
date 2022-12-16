; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; Test to confirm LAA will not find store to invariant address.
; Inner loop has no store to invariant address.
;
;  for(; i < itr; i++) {
;    for(; j < itr; j++) {
;      var2[j] = var2[j] + var1[i];
;    }
;  }

; CHECK: Non vectorizable stores to invariant address were not found in loop.
; CHECK-NOT: Non vectorizable stores to invariant address were found in loop.


define i32 @foo(ptr nocapture readonly %var1, ptr nocapture %var2, i32 %itr) #0 {
entry:
  %cmp20 = icmp eq i32 %itr, 0
  br i1 %cmp20, label %for.end10, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %for.inc8 ], [ 0, %entry ]
  %j.022 = phi i32 [ %j.1.lcssa, %for.inc8 ], [ 0, %entry ]
  %cmp218 = icmp ult i32 %j.022, %itr
  br i1 %cmp218, label %for.body3.lr.ph, label %for.inc8

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %arrayidx5 = getelementptr inbounds i32, ptr %var1, i64 %indvars.iv23
  %0 = zext i32 %j.022 to i64
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ %0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds i32, ptr %var2, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %2 = load i32, ptr %arrayidx5, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3, %for.cond1.preheader
  %j.1.lcssa = phi i32 [ %j.022, %for.cond1.preheader ], [ %itr, %for.body3 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv.next24 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %itr
  br i1 %exitcond26, label %for.end10, label %for.cond1.preheader

for.end10:                                        ; preds = %for.inc8, %entry
  ret i32 undef
}

