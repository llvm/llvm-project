; RUN: opt -passes=loop-distribute -enable-loop-distribute -S < %s | FileCheck %s

; If we can't find the bounds for one of the arrays in order to generate the
; memchecks (e.g., C[i * i] below), loop shold not get distributed.
;
;   for (i = 0; i < n; i++) {
;     A[i + 1] = A[i] * 3;
; -------------------------------
;     C[i * i] = B[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Verify that we didn't distribute by checking that we still have the original
; number of branches.

@A = common global ptr null, align 8
@B = common global ptr null, align 8
@C = common global ptr null, align 8

define void @f() {
entry:
  %a = load ptr, ptr @A, align 8
  %b = load ptr, ptr @B, align 8
  %c = load ptr, ptr @C, align 8
  br label %for.body
; CHECK: br

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %mulA = mul i32 %loadA, 3

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4

  %mulC = mul i32 %loadB, 2

  %ind_2 = mul i64 %ind, %ind
  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind_2
  store i32 %mulC, ptr %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body
; CHECK: br
; CHECK-NOT: br

for.end:                                          ; preds = %for.body
  ret void
}
