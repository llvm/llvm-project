; RUN: opt -loop-load-elim -S < %s | FileCheck %s

; Don't crash if the store and the load use different types.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     C[i] = ((float*)A)[i] * 2;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @f(
define void @f(i32* noalias %A, i32* noalias %B, i32* noalias %C, i64 %N) {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %Aidx.float = bitcast i32* %Aidx to float*

  %b = load i32, i32* %Bidx, align 4
  %a_p1 = add i32 %b, 2
  store i32 %a_p1, i32* %Aidx_next, align 4

; CHECK: %a = load float, float* %Aidx.float, align 4
  %a = load float, float* %Aidx.float, align 4
; CHECK-NEXT: %c = fmul float %a, 2.0
  %c = fmul float %a, 2.0
  %c.int = fptosi float %c to i32
  store i32 %c.int, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Don't crash if the store and the load use different types.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     A[i+1] = B[i] + 3;
;     C[i] = ((float*)A)[i] * 2;
;   }

; CHECK-LABEL: @f2(
define void @f2(i32* noalias %A, i32* noalias %B, i32* noalias %C, i64 %N) {

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1

  %Aidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %Bidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %Cidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %Aidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %Aidx.float = bitcast i32* %Aidx to float*

  %b = load i32, i32* %Bidx, align 4
  %a_p2 = add i32 %b, 2
  store i32 %a_p2, i32* %Aidx_next, align 4

  %a_p3 = add i32 %b, 3
  store i32 %a_p3, i32* %Aidx_next, align 4

; CHECK: %a = load float, float* %Aidx.float, align 4
  %a = load float, float* %Aidx.float, align 4
; CHECK-NEXT: %c = fmul float %a, 2.0
  %c = fmul float %a, 2.0
  %c.int = fptosi float %c to i32
  store i32 %c.int, i32* %Cidx, align 4

  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
