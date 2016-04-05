; RUN: opt %loadPolly -polly-detect -analyze < %s | FileCheck %s -check-prefix=DETECT
; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; DETECT: Valid Region for Scop: for.cond => for.end
; CHECK-NOT: Region: %for.cond---%for.end
;
;    void f(int A[][2]) {
;      int(*B)[2] = &A[0][0];
;      int(*C)[2] = &A[1][0];
;      for (int i = 0; i < 100; i++) {
;        B[i][0]++;
;        C[i][0]++;
;      }
;    }
;
; This test case makes sure we do not miss parts of the array subscript
; functions, which we previously did by only considering the first of a chain
; of GEP instructions. Today, we detect this situation and do not delinearize
; the relevant memory access. In the future, we may want to detect this
; pattern and combine multiple GEP functions together.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f([2 x i32]* %A) {
entry:
  %arrayidx3 = getelementptr inbounds [2 x i32], [2 x i32]* %A, i64 1, i64 0
  %tmp = bitcast i32* %arrayidx3 to [2 x i32]*
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx5 = getelementptr inbounds [2 x i32], [2 x i32]* %A, i64 %indvars.iv, i64 0
  %tmp1 = load i32, i32* %arrayidx5, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, i32* %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [2 x i32], [2 x i32]* %tmp, i64 %indvars.iv, i64 0
  %tmp2 = load i32, i32* %arrayidx8, align 4
  %inc9 = add nsw i32 %tmp2, 1
  store i32 %inc9, i32* %arrayidx8, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
