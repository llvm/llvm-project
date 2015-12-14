; RUN: opt %loadPolly -polly-scops \
; RUN:     -polly-allow-nonaffine -polly-allow-nonaffine-branches \
; RUN:     -polly-allow-nonaffine-loops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine \
; RUN:     -polly-process-unprofitable=false \
; RUN:     -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops \
; RUN:     -analyze < %s | FileCheck %s --check-prefix=PROFIT
;
; Verify that we over approximate the read acces of A[j] in the last statement as j is
; computed in a non-affine loop we do not model.
;
; CHECK:    Function: f
; CHECK:    Region: %bb2---%bb24
; CHECK:    Max Loop Depth:  1
; CHECK:    Context:
; CHECK:    [N] -> {  :
; CHECK-DAG:         N >= -2147483648
; CHECK-DAG:       and
; CHECK-DAG:         N <= 2147483647
; CHECK:           }
; CHECK:    Assumed Context:
; CHECK:    [N] -> {  :  }
; CHECK:    p0: %N
; CHECK:    Alias Groups (0):
; CHECK:        n/a
; CHECK:    Statements {
; CHECK:      Stmt_bb2
; CHECK:                [N] -> { Stmt_bb2[i0] -> MemRef_j_0__phi[] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb2[i0] -> MemRef_j_0[] };
; CHECK:      Stmt_bb4__TO__bb18
; CHECK:            Schedule :=
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> [i0, 1] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK:            MayWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_A[i0] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_smax[] };
; CHECK:            MustWriteAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_j_2__phi[] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_j_0[] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb4__TO__bb18[i0] -> MemRef_j_2__phi[] };
; CHECK:      Stmt_bb18
; CHECK:            Schedule :=
; CHECK:                [N] -> { Stmt_bb18[i0] -> [i0, 2] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb18[i0] -> MemRef_j_2[] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb18[i0] -> MemRef_j_2__phi[] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N] -> { Stmt_bb18[i0] -> MemRef_A[o0] : o0 >= -2147483648 and o0 <= 2147483645 };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N] -> { Stmt_bb18[i0] -> MemRef_A[i0] };
; CHECK:      Stmt_bb23
; CHECK:            Schedule :=
; CHECK:                [N] -> { Stmt_bb23[i0] -> [i0, 3] };
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb23[i0] -> MemRef_j_2[] };
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N] -> { Stmt_bb23[i0] -> MemRef_j_0__phi[] };
; CHECK:    }
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

define void @f(i32* %A, i32 %N, i32 %M) {
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
  %tmp5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp6 = load i32, i32* %tmp5, align 4
  %tmp7 = icmp eq i32 %tmp6, 0
  br i1 %tmp7, label %bb18, label %bb8

bb8:                                              ; preds = %bb4
  br label %bb9

bb9:                                              ; preds = %bb15, %bb8
  %j.1 = phi i32 [ 0, %bb8 ], [ %tmp16, %bb15 ]
  %tmp10 = icmp slt i32 %j.1, %M
  br i1 %tmp10, label %bb11, label %bb17

bb11:                                             ; preds = %bb9
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp13 = load i32, i32* %tmp12, align 4
  %tmp14 = add nsw i32 %tmp13, 1
  store i32 %tmp14, i32* %tmp12, align 4
  br label %bb15

bb15:                                             ; preds = %bb11
  %tmp16 = add nuw nsw i32 %j.1, 1
  br label %bb9

bb17:                                             ; preds = %bb9
  br label %bb18

bb18:                                             ; preds = %bb4, %bb17
  %j.2 = phi i32 [ %smax, %bb17 ], [ %j.0, %bb4 ]
  %tmp19 = sext i32 %j.2 to i64
  %tmp20 = getelementptr inbounds i32, i32* %A, i64 %tmp19
  %tmp21 = load i32, i32* %tmp20, align 4
  %tmp22 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp21, i32* %tmp22, align 4
  br label %bb23

bb23:                                             ; preds = %bb18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb2

bb24:                                             ; preds = %bb2
  ret void
}
