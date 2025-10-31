; RUN: opt %loadNPMPolly -polly-detect-full-functions -polly-invariant-load-hoisting '-passes=cgscc(polly-inline),function(print<polly-function-scops>)' -disable-output < %s 2>&1 | FileCheck %s

; Check that we inline a function that requires invariant load hoisting
; correctly.
; CHECK:    Max Loop Depth:  2


; void to_be_inlined(int A[], int *begin, int *end) {
;     for(int i = *begin; i < *end; i++) {
;         A[i] = 10;
;     }
; }
;
; static const int N = 1000;
;
; void inline_site(int A[], int *begin, int *end) {
;     for(int i = 0; i < N; i++)
;         to_be_inlined(A);
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @to_be_inlined(ptr %A, ptr %begin, ptr %end) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = load i32, ptr %begin, align 4
  %tmp21 = load i32, ptr %end, align 4
  %cmp3 = icmp slt i32 %tmp, %tmp21
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  %tmp1 = sext i32 %tmp to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv4 = phi i64 [ %tmp1, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv4
  store i32 10, ptr %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv4, 1
  %tmp2 = load i32, ptr %end, align 4
  %tmp3 = sext i32 %tmp2 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %tmp3
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}


define void @inline_site(ptr %A, ptr %begin, ptr %end) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %i.01 = phi i32 [ 0, %entry.split ], [ %inc, %for.body ]
  tail call void @to_be_inlined(ptr %A, ptr %begin, ptr %end)
  %inc = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

