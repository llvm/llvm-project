; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s
; RUN: opt -passes=loop-load-elim -S -runtime-check-per-loop-load-elim=2 < %s | FileCheck %s --check-prefix=AGGRESSIVE

; This needs two pairs of memchecks (A * { C, D }) for a single load
; elimination which is considered to expansive by default.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     C[i] = A[i] * 2;
;     D[i] = 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr  %A, ptr  %B, ptr  %C, i64 %N, ptr %D) {
entry:
  br label %for.body

; AGGRESSIVE: for.body.lver.check:
; AGGRESSIVE: %found.conflict{{.*}} =
; AGGRESSIVE: %found.conflict{{.*}} =
; AGGRESSIVE-NOT: %found.conflict{{.*}} =

for.body:                                         ; preds = %for.body, %entry
; CHECK-NOT: %store_forwarded =
; AGGRESSIVE: %store_forwarded =
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %Didx = getelementptr inbounds i32, ptr %D, i64 %indvars.iv

  %b = load i32, ptr %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, ptr %Aidx_next, align 4

  %a = load i32, ptr %Aidx, align 4
; CHECK: %c = mul i32 %a, 2
; AGGRESSIVE: %c = mul i32 %store_forwarded, 2
  %c = mul i32 %a, 2
  store i32 %c, ptr %Cidx, align 4
  store i32 2, ptr %Didx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
