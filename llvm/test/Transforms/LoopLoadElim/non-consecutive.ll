; RUN: opt -loop-load-elim -S < %s | FileCheck %s

; The accesses to A are independent here but LAA reports it as a loop-carried
; forward dependence.  Check that we don't perform st->ld forwarding between
; them.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i][1] = B[i] + 2;
;     C[i] = A[i][0] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f([2 x i32]* noalias %A, i32* noalias %B, i32* noalias %C, i64 %N) {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %A1idx = getelementptr inbounds [2 x i32], [2 x i32]* %A, i64 %indvars.iv, i32 1
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %A0idx = getelementptr inbounds [2 x i32], [2 x i32]* %A, i64 %indvars.iv, i32 0

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %A1idx, align 4

; CHECK: %a = load i32, i32* %A0idx, align 4
  %a = load i32, i32* %A0idx, align 4
; CHECK: %c = mul i32 %a, 2
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
