; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; Check that loop-indepedent forward dependences are discovered properly.
;
; FIXME: This does not actually always work which is pretty confusing.  Right
; now there is hack in LAA that tries to figure out loop-indepedent forward
; dependeces *outside* of the MemoryDepChecker logic (i.e. proper dependence
; analysis).
;
; Therefore if there is only loop-independent dependences for an array
; (i.e. the same index is used), we don't discover the forward dependence.
; So, at ***, we add another non-I-based access of A to trigger
; MemoryDepChecker analysis for accesses of A.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i + 1] = B[i] + 1;   // ***
;     A[i] = B[i] + 2;
;     C[i] = A[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias %A, i32* noalias %B, i32* noalias %C, i64 %N) {

; CHECK: Dependences:
; CHECK-NEXT:   Forward:
; CHECK-NEXT:       store i32 %b_p1, i32* %Aidx, align 4 ->
; CHECK-NEXT:       %a = load i32, i32* %Aidx, align 4
; CHECK:        ForwardButPreventsForwarding:
; CHECK-NEXT:       store i32 %b_p2, i32* %Aidx_next, align 4 ->
; CHECK-NEXT:       %a = load i32, i32* %Aidx, align 4
; CHECK:        Forward:
; CHECK-NEXT:       store i32 %b_p2, i32* %Aidx_next, align 4 ->
; CHECK-NEXT:       store i32 %b_p1, i32* %Aidx, align 4

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv

  %b = load i32, i32* %Bidx, align 4
  %b_p2 = add i32 %b, 1
  store i32 %b_p2, i32* %Aidx_next, align 4

  %b_p1 = add i32 %b, 2
  store i32 %b_p1, i32* %Aidx, align 4

  %a = load i32, i32* %Aidx, align 4
  %c = mul i32 %a, 2
  store i32 %c, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
