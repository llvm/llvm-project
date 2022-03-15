; RUN: opt %loadPolly -polly-detect-full-functions -polly-scop-inliner \
; RUN: -polly-print-scops -disable-output < %s | FileCheck %s

; Check that we get the 2 nested loops by inlining `to_be_inlined` into
; `inline_site`.
; CHECK:    Max Loop Depth:  2

; static const int N = 1000;
;
; void to_be_inlined(int A[]) {
;     for(int i = 0; i < N; i++)
;         A[i] *= 10;
; }
;
; void inline_site(int A[]) {
;     for(int i = 0; i < N; i++)
;         to_be_inlined(A);
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"


define void @to_be_inlined(i32* %A) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv1 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %tmp, 10
  store i32 %mul, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @inline_site(i32* %A) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %i.01 = phi i32 [ 0, %entry.split ], [ %inc, %for.body ]
  tail call void @to_be_inlined(i32* %A)
  %inc = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

