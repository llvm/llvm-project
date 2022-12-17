; RUN: opt -S -passes=loop-load-elim < %s | FileCheck %s

; We can't hoist conditional loads to the preheader for the initial value.
; E.g. in the loop below we'd access array[-1] if we did:
;
;   for(int i = 0 ; i < n ; i++ )
;     array[i] = ( i > 0 ? array[i - 1] : 0 ) + 4;

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @f(i32* %array, i32 %n) {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %cond.end, %entry
  ret void

for.body:                                         ; preds = %entry, %cond.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %cond.end ], [ 0, %entry ]
; CHECK-NOT: %store_forwarded = phi
  %cmp1 = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp1, label %cond.true, label %cond.end

cond.true:                                        ; preds = %for.body
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds i32, i32* %array, i64 %0
  %1 = load i32, i32* %arrayidx, align 4
  br label %cond.end

cond.end:                                         ; preds = %for.body, %cond.true
  %cond = phi i32 [ %1, %cond.true ], [ 0, %for.body ]
; CHECK: %cond = phi i32 [ %1, %cond.true ], [ 0, %for.body ]
  %add = add nsw i32 %cond, 4
  %arrayidx3 = getelementptr inbounds i32, i32* %array, i64 %indvars.iv
  store i32 %add, i32* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
