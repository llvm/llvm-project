; RUN: opt -basicaa -loop-load-elim -S < %s | FileCheck %s

; When optimizing for size don't eliminate in this loop because the loop would
; have to be versioned first because A and C may alias.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     C[i] = A[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @f(
define void @f(i32* %A, i32* %B, i32* %C, i64 %N) optsize {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_next, align 4

  %a = load i32, i32* %Aidx, align 4
; CHECK: %c = mul i32 %a, 2
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Same loop but with noalias on %A and %C.  In this case load-eliminate even
; with -Os.

; CHECK-LABEL: @g(
define void @g(i32* noalias %A, i32* %B, i32* noalias %C, i64 %N) optsize {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_next, align 4

  %a = load i32, i32* %Aidx, align 4
; CHECK: %c = mul i32 %store_forwarded, 2
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
