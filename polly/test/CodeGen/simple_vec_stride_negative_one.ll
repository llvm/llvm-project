; RUN: opt %loadPolly -polly-codegen -polly-vectorizer=polly -S < %s | FileCheck %s

; ModuleID = 'reverse.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

;int A[100];
;void foo() {
;  for (int i=3; i >= 0; i--)
;    A[i]+=1;
;}


@A = common global [100 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 3, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %arrayidx, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %1 = trunc i64 %indvars.iv to i32
  %cmp = icmp sgt i32 %1, 0
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK: @foo
; CHECK: [[LOAD:%[a-zA-Z0-9_]+]] = load <4 x i32>, <4 x i32>*
; CHECK: [[REVERSE_LOAD:%[a-zA-Z0-9_]+reverse]] = shufflevector <4 x i32> [[LOAD]], <4 x i32> [[LOAD]], <4 x i32> <i32 3, i32 2, i32 1, i32 0>
