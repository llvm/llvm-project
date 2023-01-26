; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-function-scops -disable-output < %s | FileCheck %s
;
; CHECK: Reduction Type: NONE
;
; Check that we do not mark these accesses as reduction like.
; We do this for the case the loads are modelt with the same LLVM-IR value and
; for the case there are different LLVM-IR values.
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = A[i] + A[i];
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f_one_load_case(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, %tmp
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %i.0
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @f_two_loads_case(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidxCopy = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmpCopy = load i32, ptr %arrayidxCopy, align 4
  %add = add nsw i32 %tmp, %tmpCopy
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %i.0
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
