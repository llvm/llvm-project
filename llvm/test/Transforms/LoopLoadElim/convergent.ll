; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

; Simple st->ld forwarding derived from a lexical forward dep, but
; can't introduce runtime checks with a convergent call.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = convergent(B[i]) + 2;
;     C[i] = A[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @f_convergent(
; CHECK: call i32 @llvm.convergent(
; CHECK-NOT: call i32 @llvm.convergent(
define void @f_convergent(ptr %A, ptr %B, ptr %C, i64 %N) #0 {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv

  %b = load i32, ptr %Bidx, align 4
  %convergent.b = call i32 @llvm.convergent(i32 %b)

  %a_p1 = add i32 %convergent.b, 2
  store i32 %a_p1, ptr %Aidx_next, align 4

  %a = load i32, ptr %Aidx, align 1
  %c = mul i32 %a, 2
  store i32 %c, ptr %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare i32 @llvm.convergent(i32) #1

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind readnone convergent }
