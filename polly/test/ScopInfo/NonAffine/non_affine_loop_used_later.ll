; RUN: opt %loadNPMPolly -polly-allow-nonaffine -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops                                                                        '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly -polly-allow-nonaffine -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops -polly-unprofitable-scalar-accs=true -polly-process-unprofitable=false '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=PROFIT
;
; Verify that we over approximate the read acces of A[j] in the last statement as j is
; computed in a non-affine loop we do not model.
;
; CHECK:      Function: f
; CHECK-NEXT: Region: %bb2---%bb24
; CHECK-NEXT: Max Loop Depth:  1
; CHECK-NEXT: Invariant Accesses: {
; CHECK-NEXT: }
; CHECK-NEXT: Context:
; CHECK-NEXT: [N] -> {  : -2147483648 <= N <= 2147483647 }
; CHECK-NEXT: Assumed Context:
; CHECK-NEXT: [N] -> {  :  }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT: [N] -> {  : false }
; CHECK:      p0: %N
; CHECK-NEXT: Arrays {
; CHECK-NEXT:     i32 MemRef_j_0__phi; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_0; // Element size 4
; CHECK-NEXT:     i32 MemRef_A[*]; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_2__phi; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_2; // Element size 4
; CHECK-NEXT: }
; CHECK-NEXT: Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     i32 MemRef_j_0__phi; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_0; // Element size 4
; CHECK-NEXT:     i32 MemRef_A[*]; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_2__phi; // Element size 4
; CHECK-NEXT:     i32 MemRef_j_2; // Element size 4
; CHECK-NEXT: }
; CHECK-NEXT: Alias Groups (0):
; CHECK-NEXT:     n/a
; CHECK-NEXT: Statements {
; CHECK-NEXT:     Stmt_bb2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb2[i0] : 0 <= i0 <= N; Stmt_bb2[0] : N < 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb2[i0] -> [i0, 0] : i0 <= N; Stmt_bb2[0] -> [0, 0] : N < 0 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb2[i0] -> MemRef_j_0__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb2[i0] -> MemRef_j_0[] };
; CHECK-NEXT:     Stmt_bb4__TO__bb18
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> [i0, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_j_2__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_j_0[] };
; CHECK-NEXT:     Stmt_bb18
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] -> [i0, 2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] -> MemRef_j_2[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] -> MemRef_j_2__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] -> MemRef_A[o0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb18[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_bb23
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb23[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb23[i0] -> [i0, 3] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb23[i0] -> MemRef_j_2[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_bb23[i0] -> MemRef_j_0__phi[] };
; CHECK-NEXT: }
;
; Due to the scalar accesses we are not able to distribute the outer loop, thus we do not consider the region profitable.
;
; PROFIT-NOT: Statements
;
;    void f(int *A, int N, int M) {
;      int i = 0, j = 0;
;      for (i = 0; i < N; i++) {
;        if (A[i])
;          for (j = 0; j < M; j++)
;            A[i]++;
;        A[i] = A[j];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %N, i32 %M) {
bb:
  %tmp = icmp sgt i32 %M, 0
  %smax = select i1 %tmp, i32 %M, i32 0
  %tmp1 = sext i32 %N to i64
  br label %bb2

bb2:                                              ; preds = %bb23, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb23 ], [ 0, %bb ]
  %j.0 = phi i32 [ 0, %bb ], [ %j.2, %bb23 ]
  %tmp3 = icmp slt i64 %indvars.iv, %tmp1
  br i1 %tmp3, label %bb4, label %bb24

bb4:                                              ; preds = %bb2
  %tmp5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp6 = load i32, ptr %tmp5, align 4
  %tmp7 = icmp eq i32 %tmp6, 0
  br i1 %tmp7, label %bb18, label %bb8

bb8:                                              ; preds = %bb4
  br label %bb9

bb9:                                              ; preds = %bb15, %bb8
  %j.1 = phi i32 [ 0, %bb8 ], [ %tmp16, %bb15 ]
  %tmp10 = icmp slt i32 %j.1, %M
  br i1 %tmp10, label %bb11, label %bb17

bb11:                                             ; preds = %bb9
  %tmp12 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp13 = load i32, ptr %tmp12, align 4
  %tmp14 = add nsw i32 %tmp13, 1
  store i32 %tmp14, ptr %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb11
  %tmp16 = add nuw nsw i32 %j.1, 1
  br label %bb9

bb17:                                             ; preds = %bb9
  br label %bb18

bb18:                                             ; preds = %bb4, %bb17
  %j.2 = phi i32 [ %smax, %bb17 ], [ %j.0, %bb4 ]
  %tmp19 = sext i32 %j.2 to i64
  %tmp20 = getelementptr inbounds i32, ptr %A, i64 %tmp19
  %tmp21 = load i32, ptr %tmp20, align 4
  %tmp22 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp21, ptr %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb2

bb24:                                             ; preds = %bb2
  ret void
}
