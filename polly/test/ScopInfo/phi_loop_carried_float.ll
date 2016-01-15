; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    float f(float *A, int N) {
;      float tmp = 0;
;      for (int i = 0; i < N; i++)
;        tmp += A[i];
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb1[i0] : i0 >= 0 and i0 <= N; Stmt_bb1[0] : N <= -1 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb1[i0] -> [i0, 0] : i0 <= N; Stmt_bb1[0] -> [0, 0] : N <= -1 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb1[i0] -> MemRef_tmp_0__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb1[i0] -> MemRef_tmp_0[] };
; CHECK-NEXT:     Stmt_bb4
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb4[i0] : i0 <= -1 + N and i0 >= 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb4[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb4[i0] -> MemRef_tmp_0__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb4[i0] -> MemRef_tmp_0[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb4[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(float* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb1

bb1:                                              ; preds = %bb4, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb4 ], [ 0, %bb ]
  %tmp.0 = phi float [ 0.000000e+00, %bb ], [ %tmp7, %bb4 ]
  %tmp2 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp2, label %bb3, label %bb8

bb3:                                              ; preds = %bb1
  br label %bb4

bb4:                                              ; preds = %bb3
  %tmp5 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fadd float %tmp.0, %tmp6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  br label %exit

exit:
  ret void
}
