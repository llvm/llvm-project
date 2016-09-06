; RUN: opt -loop-accesses -analyze < %s | FileCheck %s
; RUN: opt -passes='require<scalar-evolution>,require<aa>,loop(print-access-info)' -disable-output  < %s 2>&1 | FileCheck %s

;   for (unsigned i = 0; i < 100; i++) {
;     A[i+8] = B[i] + 2;
;     C[i] = A[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %B, i32* %C, i64 %N) {

; CHECK: Dependences:
; CHECK-NEXT: Forward:
; CHECK-NEXT:   store i32 %a_p1, i32* %Aidx_ahead, align 4 ->
; CHECK-NEXT:   %a = load i32, i32* %Aidx, align 4

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %idx = add nuw nsw i64 %indvars.iv, 8

  %Aidx_ahead = getelementptr inbounds i32, i32* %A, i64 %idx
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_ahead, align 4

  %a = load i32, i32* %Aidx, align 4
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
