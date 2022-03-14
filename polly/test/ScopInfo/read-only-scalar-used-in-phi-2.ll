; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    float foo(float sum, float A[]) {
;
;      for (long i = 0; i < 100; i++)
;        sum += A[i];
;
;      return sum;
;    }

; Verify that we do not model the read from %sum. Reads that only happen in
; case control flow reaches the PHI node from outside the SCoP are handled
; implicitly during code generation.

; CHECK: Stmt_bb1[i0] -> MemRef_phisum__ph
; CHECK-NOT: Stmt_bb1[i0] -> MemRef_sum[]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float %sum, float* %A) {
bb:
  br label %bb1

bb1:
  %i = phi i64 [ 0, %bb ], [ %i.next, %bb1 ]
  %phisum = phi float [ %sum, %bb ], [ %tmp5, %bb1 ]
  %tmp = getelementptr inbounds float, float* %A, i64 %i
  %tmp4 = load float, float* %tmp, align 4
  %tmp5 = fadd float %phisum, %tmp4
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i, 100
  br i1 %exitcond, label %bb1, label %bb7

bb7:
  ret float %phisum
}
