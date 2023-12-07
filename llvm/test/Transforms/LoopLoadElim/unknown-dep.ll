; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Give up in the presence of unknown deps. Here, the different strides result
; in unknown dependence:
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     A[2*i] = C[i] + 2;
;     D[i] = A[i] + 2;
;   }

define void @f(ptr noalias %A, ptr noalias %B, ptr noalias %C,
               ptr noalias %D, i64 %N) {

entry:
; for.body.ph:
; CHECK-NOT: %load_initial =
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK-NOT: %store_forwarded =
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %Didx = getelementptr inbounds i32, ptr %D, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %indvars.m2 = mul nuw nsw i64 %indvars.iv, 2
  %A2idx = getelementptr inbounds i32, ptr %A, i64 %indvars.m2

  %b = load i32, ptr %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, ptr %Aidx_next, align 4

  %c = load i32, ptr %Cidx, align 4
  %a_m2 = add i32 %c, 2
  store i32 %a_m2, ptr %A2idx, align 4

  %a = load i32, ptr %Aidx, align 4
; CHECK-NOT: %d = add i32 %store_forwarded, 2
; CHECK: %d = add i32 %a, 2
  %d = add i32 %a, 2
  store i32 %d, ptr %Didx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
