; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    #define N 400
;
;    void first_higher_dimensional(float A[][N]) {
;      for (long i = 0; i < N; i++)
;        for (long j = 0; j < N; j++)
;          A[i][j] += i + j;
;
;      A[0][0] += A[100][100];
;
;      for (long i = 0; i < N; i++)
;        for (long j = 0; j < N; j++)
;          A[i][j] += i + j;
;    }

;    void first_lower_dimensional(float A[][N], float B[][N]) {
;      for (long i = 0; i < N; i++)
;        for (long j = 0; j < N; j++)
;          B[i][j] += i + j;
;
;      A[0][0] += B[100][100];
;
;      for (long i = 0; i < N; i++)
;        for (long j = 0; j < N; j++)
;          A[i][j] += i + j;
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] : 0 <= i0 <= 399 and 0 <= i1 <= 399 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> [0, i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:     Stmt_bb17
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb17[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb17[] -> [1, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_A[100, 100] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_A[0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_A[0, 0] };
; CHECK-NEXT:     Stmt_bb26
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb26[i0, i1] : 0 <= i0 <= 399 and 0 <= i1 <= 399 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> [2, i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] : 0 <= i0 <= 399 and 0 <= i1 <= 399 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> [0, i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_B[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_B[i0, i1] };
; CHECK-NEXT:     Stmt_bb17
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb17[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb17[] -> [1, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_B[100, 100] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_A[0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb17[] -> MemRef_A[0, 0] };
; CHECK-NEXT:     Stmt_bb26
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb26[i0, i1] : 0 <= i0 <= 399 and 0 <= i1 <= 399 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> [2, i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb26[i0, i1] -> MemRef_A[i0, i1] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @first_higher_dimensional(ptr %A) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb15, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp16, %bb15 ]
  %exitcond3 = icmp ne i64 %i.0, 400
  br i1 %exitcond3, label %bb5, label %bb17

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb12, %bb5
  %j.0 = phi i64 [ 0, %bb5 ], [ %tmp13, %bb12 ]
  %exitcond2 = icmp ne i64 %j.0, 400
  br i1 %exitcond2, label %bb7, label %bb14

bb7:                                              ; preds = %bb6
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp8 = sitofp i64 %tmp to float
  %tmp9 = getelementptr inbounds [400 x float], ptr %A, i64 %i.0, i64 %j.0
  %tmp10 = load float, ptr %tmp9, align 4
  %tmp11 = fadd float %tmp10, %tmp8
  store float %tmp11, ptr %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb7
  %tmp13 = add nuw nsw i64 %j.0, 1
  br label %bb6

bb14:                                             ; preds = %bb6
  br label %bb15

bb15:                                             ; preds = %bb14
  %tmp16 = add nuw nsw i64 %i.0, 1
  br label %bb4

bb17:                                             ; preds = %bb4
  %tmp18 = getelementptr inbounds [400 x float], ptr %A, i64 100, i64 100
  %tmp19 = load float, ptr %tmp18, align 4
  %tmp21 = load float, ptr %A, align 4
  %tmp22 = fadd float %tmp21, %tmp19
  store float %tmp22, ptr %A, align 4
  br label %bb23

bb23:                                             ; preds = %bb35, %bb17
  %i1.0 = phi i64 [ 0, %bb17 ], [ %tmp36, %bb35 ]
  %exitcond1 = icmp ne i64 %i1.0, 400
  br i1 %exitcond1, label %bb24, label %bb37

bb24:                                             ; preds = %bb23
  br label %bb25

bb25:                                             ; preds = %bb32, %bb24
  %j2.0 = phi i64 [ 0, %bb24 ], [ %tmp33, %bb32 ]
  %exitcond = icmp ne i64 %j2.0, 400
  br i1 %exitcond, label %bb26, label %bb34

bb26:                                             ; preds = %bb25
  %tmp27 = add nuw nsw i64 %i1.0, %j2.0
  %tmp28 = sitofp i64 %tmp27 to float
  %tmp29 = getelementptr inbounds [400 x float], ptr %A, i64 %i1.0, i64 %j2.0
  %tmp30 = load float, ptr %tmp29, align 4
  %tmp31 = fadd float %tmp30, %tmp28
  store float %tmp31, ptr %tmp29, align 4
  br label %bb32

bb32:                                             ; preds = %bb26
  %tmp33 = add nuw nsw i64 %j2.0, 1
  br label %bb25

bb34:                                             ; preds = %bb25
  br label %bb35

bb35:                                             ; preds = %bb34
  %tmp36 = add nuw nsw i64 %i1.0, 1
  br label %bb23

bb37:                                             ; preds = %bb23
  ret void
}

define void @first_lower_dimensional(ptr %A, ptr %B) {
bb:
  br label %bb4

bb4:                                              ; preds = %bb15, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp16, %bb15 ]
  %exitcond3 = icmp ne i64 %i.0, 400
  br i1 %exitcond3, label %bb5, label %bb17

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb12, %bb5
  %j.0 = phi i64 [ 0, %bb5 ], [ %tmp13, %bb12 ]
  %exitcond2 = icmp ne i64 %j.0, 400
  br i1 %exitcond2, label %bb7, label %bb14

bb7:                                              ; preds = %bb6
  %tmp = add nuw nsw i64 %i.0, %j.0
  %tmp8 = sitofp i64 %tmp to float
  %tmp9 = getelementptr inbounds [400 x float], ptr %B, i64 %i.0, i64 %j.0
  %tmp10 = load float, ptr %tmp9, align 4
  %tmp11 = fadd float %tmp10, %tmp8
  store float %tmp11, ptr %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb7
  %tmp13 = add nuw nsw i64 %j.0, 1
  br label %bb6

bb14:                                             ; preds = %bb6
  br label %bb15

bb15:                                             ; preds = %bb14
  %tmp16 = add nuw nsw i64 %i.0, 1
  br label %bb4

bb17:                                             ; preds = %bb4
  %tmp18 = getelementptr inbounds [400 x float], ptr %B, i64 100, i64 100
  %tmp19 = load float, ptr %tmp18, align 4
  %tmp21 = load float, ptr %A, align 4
  %tmp22 = fadd float %tmp21, %tmp19
  store float %tmp22, ptr %A, align 4
  br label %bb23

bb23:                                             ; preds = %bb35, %bb17
  %i1.0 = phi i64 [ 0, %bb17 ], [ %tmp36, %bb35 ]
  %exitcond1 = icmp ne i64 %i1.0, 400
  br i1 %exitcond1, label %bb24, label %bb37

bb24:                                             ; preds = %bb23
  br label %bb25

bb25:                                             ; preds = %bb32, %bb24
  %j2.0 = phi i64 [ 0, %bb24 ], [ %tmp33, %bb32 ]
  %exitcond = icmp ne i64 %j2.0, 400
  br i1 %exitcond, label %bb26, label %bb34

bb26:                                             ; preds = %bb25
  %tmp27 = add nuw nsw i64 %i1.0, %j2.0
  %tmp28 = sitofp i64 %tmp27 to float
  %tmp29 = getelementptr inbounds [400 x float], ptr %A, i64 %i1.0, i64 %j2.0
  %tmp30 = load float, ptr %tmp29, align 4
  %tmp31 = fadd float %tmp30, %tmp28
  store float %tmp31, ptr %tmp29, align 4
  br label %bb32

bb32:                                             ; preds = %bb26
  %tmp33 = add nuw nsw i64 %j2.0, 1
  br label %bb25

bb34:                                             ; preds = %bb25
  br label %bb35

bb35:                                             ; preds = %bb34
  %tmp36 = add nuw nsw i64 %i1.0, 1
  br label %bb23

bb37:                                             ; preds = %bb23
  ret void
}
