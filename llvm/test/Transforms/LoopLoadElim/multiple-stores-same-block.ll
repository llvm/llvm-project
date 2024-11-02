; RUN: opt -loop-load-elim -S < %s | FileCheck %s

; In this case the later store forward to the load:
;
;   for (unsigned i = 0; i < 100; i++) {
;     B[i] = A[i] + 1;
;     A[i+1] = C[i] + 2;
;     A[i+1] = D[i] + 3;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias nocapture %A, i32* noalias nocapture readonly %B,
               i32* noalias nocapture %C, i32* noalias nocapture readonly %D,
               i64 %N) {
entry:
; CHECK: %load_initial = load i32, i32* %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK: %store_forwarded = phi i32 [ %load_initial, %entry ], [ %addD, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidxA = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %loadA = load i32, i32* %arrayidxA, align 4
; CHECK: %addA = add i32 %store_forwarded, 1
  %addA = add i32 %loadA, 1

  %arrayidxB = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %addA, i32* %arrayidxB, align 4

  %arrayidxC = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %loadC = load i32, i32* %arrayidxC, align 4
  %addC = add i32 %loadC, 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidxA_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  store i32 %addC, i32* %arrayidxA_next, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %D, i64 %indvars.iv
  %loadD = load i32, i32* %arrayidxD, align 4
  %addD = add i32 %loadD, 3
  store i32 %addD, i32* %arrayidxA_next, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
