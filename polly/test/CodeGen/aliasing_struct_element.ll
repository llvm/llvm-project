; RUN: opt %loadNPMPolly -S -passes=polly-codegen < %s | FileCheck %s
;
; We should only access (or compute the address of) "the first element" of %S
; as it is a single struct not a struct array. The maximal access to S, thus
; S->B[1023] is for ScalarEvolution an access with offset of 1423, 1023 for the
; index inside the B part of S and 400 to skip the Dummy array in S. Note that
; these numbers are relative to the actual type of &S->B[i] (char*) not to the
; type of S (struct st *) or something else.
;
; Verify that we do not use the offset 1423 into a non existent S array when we
; compute runtime alias checks but treat it as if it was a char array.
;
; CHECK: %polly.access.S = getelementptr i8, ptr %S, i64 1424
;
;    struct st {
;      int Dummy[100];
;      char B[100];
;    };
;
;    void jd(int *A, struct st *S) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = S->B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.st = type { [100 x i32], [100 x i8] }

define void @jd(ptr %A, ptr %S) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds %struct.st, ptr %S, i64 0, i32 1, i64 %indvars.iv
  %tmp = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %tmp to i32
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %conv, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
