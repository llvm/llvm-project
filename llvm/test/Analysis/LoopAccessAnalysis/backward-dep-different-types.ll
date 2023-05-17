; RUN: opt -passes='print<access-info>' -disable-output < %s  2>&1 | FileCheck %s

; In this loop just because we access A through different types (int, float)
; we still have a dependence cycle:
;
;   for (i = 0; i < n; i++) {
;    A_float = (float *) A;
;    A_float[i + 1] = A[i] * B[i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK: Report: unsafe dependent memory operations in loop
; CHECK-NOT: Memory dependences are safe

@B = common global ptr null, align 8
@A = common global ptr null, align 8

define void @f() {
entry:
  %a = load ptr, ptr @A, align 8
  %b = load ptr, ptr @B, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %storemerge3 = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %storemerge3
  %loadA = load i32, ptr %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %storemerge3
  %loadB = load i32, ptr %arrayidxB, align 2

  %mul = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %storemerge3, 1

  %arrayidxA_plus_2 = getelementptr inbounds float, ptr %a, i64 %add
  %mul_float = sitofp i32 %mul to float
  store float %mul_float, ptr %arrayidxA_plus_2, align 2

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
