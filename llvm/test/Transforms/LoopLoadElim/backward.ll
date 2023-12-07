; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

; Simple st->ld forwarding derived from a lexical backward dep.
;
;   for (unsigned i = 0; i < 100; i++)
;     A[i+1] = A[i] + B[i];

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, i64 %N) {
entry:
; CHECK: %load_initial = load i32, ptr %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK: %store_forwarded = phi i32 [ %load_initial, %entry ], [ %add, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %load = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %load_1 = load i32, ptr %arrayidx2, align 4
; CHECK: %add = add i32 %load_1, %store_forwarded
  %add = add i32 %load_1, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next
  store i32 %add, ptr %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Same but loop is descending.
;
;   for (unsigned i = N; i > 0; i--)
;     A[i-1] = A[i] + B[i];
define void @g(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, i64 %N) {
entry:
; CHECK: %0 = shl i64 %N, 2
; CHECK: %scevgep = getelementptr i8, ptr %A, i64 %0
; CHECK: %load_initial = load i32, ptr %scevgep, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK: %store_forwarded = phi i32 [ %load_initial, %entry ], [ %add, %for.body ]
  %i.09 = phi i64 [ %sub, %for.body ], [ %N, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i.09
  %load = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i64 %i.09
  %load_1 = load i32, ptr %arrayidx1, align 4
; CHECK: %add = add i32 %load_1, %store_forwarded
  %add = add i32 %load_1, %load
  %sub = add i64 %i.09, -1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %sub
  store i32 %add, ptr %arrayidx2, align 4
  %cmp.not = icmp eq i64 %sub, 0
  br i1 %cmp.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

