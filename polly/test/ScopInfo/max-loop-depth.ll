; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void bar();
;    void foo(int *A, int *B, long int N, long int M) {
;      for (long int j = 0; j < M; ++j) {
;        bar();
;        for (long int i = 0; i < N; ++i)
;          A[i] += 1;
;        for (long int i = 0; i < N; ++i)
;          A[i] += 1;
;      }
;    }
;
; Test to check that the scop only counts loop depth for loops fully contained
; in the scop.
; CHECK: Max Loop Depth: 1
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B, i64 %N, i64 %M) {
entry:
  %cmp1 = icmp slt i64 0, %M
  br i1 %cmp1, label %for.body1, label %for.end1

  for.body1:                                         ; preds = %entry, %for.inc1
    %j.0 = phi i64 [ 0, %entry ], [ %j.next, %for.inc1 ]
    call void (...) @bar() #0
    %cmp2 = icmp slt i64 0, %N
    br i1 %cmp2, label %for.body2, label %for.end2

  for.body2:                                        ; preds = %for.body1, %for.inc2
    %i.1 = phi i64 [ 0, %for.body1 ], [ %i.next.1, %for.inc2 ]
    %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i.1
    %tmp = load i32, ptr %arrayidx, align 4
    %add = add nsw i32 %tmp, 1
    store i32 %add, ptr %arrayidx, align 4
    br label %for.inc2

  for.inc2:                                          ; preds = %for.body2
    %i.next.1 = add nuw nsw i64 %i.1, 1
    %cmp3 = icmp slt i64 %i.next.1, %N
    br i1 %cmp3, label %for.body2, label %for.end2


  for.end2:                                          ; preds = %for.inc2, %for.body1
    %cmp4 = icmp slt i64 0, %N
    br i1 %cmp4, label %for.body3, label %for.end3

  for.body3:					   ; preds = %for.end2
    %i.2 = phi i64 [ 0, %for.end2 ], [ %i.next.2, %for.inc3 ]
    %arrayidx1 = getelementptr inbounds i32, ptr %A, i64 %i.2
    %tmp1 = load i32, ptr %arrayidx1, align 4
    %add1 = add nsw i32 %tmp1, 1
    store i32 %add1, ptr %arrayidx1, align 4
    br label %for.inc3

  for.inc3:					  ; preds = %for.body3
    %i.next.2 = add nuw nsw i64 %i.2, 1
    %cmp5 = icmp slt i64 %i.next.2, %N
    br i1 %cmp5, label %for.body3, label %for.end3

  for.end3:					  ; preds = %for.inc3, %for.end2
    br label %for.inc1

  for.inc1:					  ; preds = %for.end3
    %j.next = add nuw nsw i64 %j.0, 1
    %cmp6 = icmp slt i64 %j.next, %M
    br i1 %cmp6, label %for.body1, label %for.end1

  for.end1:                                        ; preds = %entry, %for.inc1
    ret void
  }

declare void @bar(...) #0
