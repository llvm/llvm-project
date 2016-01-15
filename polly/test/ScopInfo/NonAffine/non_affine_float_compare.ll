; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine-branches \
; RUN:                -analyze < %s | FileCheck %s
;
;    void f(float *A) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i] == A[i - 1])
;          A[i]++;
;    }
;
; CHECK:      Function: f
; CHECK-NEXT: Region: %bb1---%bb14
; CHECK-NEXT: Max Loop Depth:  1
; CHECK-NEXT: Invariant Accesses: {
; CHECK-NEXT: }
; CHECK-NEXT: Context:
; CHECK-NEXT: {  :  }
; CHECK-NEXT: Assumed Context:
; CHECK-NEXT: {  :  }
; CHECK-NEXT: Boundary Context:
; CHECK-NEXT: {  :  }
; CHECK-NEXT: Arrays {
; CHECK-NEXT:     float MemRef_A[*]; // Element size 4
; CHECK-NEXT: }
; CHECK-NEXT: Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     float MemRef_A[*]; // Element size 4
; CHECK-NEXT: }
; CHECK-NEXT: Alias Groups (0):
; CHECK-NEXT:     n/a
; CHECK-NEXT: Statements {
; CHECK-NEXT:     Stmt_bb2__TO__bb12
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] -> MemRef_A[-1 + i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb2__TO__bb12[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(float* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb13, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb13 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb14

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp3 = load float, float* %tmp, align 4
  %tmp4 = add nsw i64 %indvars.iv, -1
  %tmp5 = getelementptr inbounds float, float* %A, i64 %tmp4
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fcmp oeq float %tmp3, %tmp6
  br i1 %tmp7, label %bb8, label %bb12

bb8:                                              ; preds = %bb2
  %tmp9 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp10 = load float, float* %tmp9, align 4
  %tmp11 = fadd float %tmp10, 1.000000e+00
  store float %tmp11, float* %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb8, %bb2
  br label %bb13

bb13:                                             ; preds = %bb12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb14:                                             ; preds = %bb1
  ret void
}
