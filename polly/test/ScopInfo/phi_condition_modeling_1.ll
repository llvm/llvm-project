; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void f(int *A, int c, int N) {
;      int tmp;
;      for (int i = 0; i < N; i++) {
;        if (i > c)
;          tmp = 3;
;        else
;          tmp = 5;
;        A[i] = tmp;
;      }
;    }
;
; CHECK:    Statements {
; CHECK-LABEL:      Stmt_bb6
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N, c] -> { Stmt_bb6[i0] -> MemRef_tmp_0__phi[] };
; CHECK-NOT: Access
; CHECK-LABEL:      Stmt_bb7
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N, c] -> { Stmt_bb7[i0] -> MemRef_tmp_0__phi[] };
; CHECK-NOT: Access
; CHECK-LABEL:      Stmt_bb8
; CHECK-NOT: Access
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [N, c] -> { Stmt_bb8[i0] -> MemRef_tmp_0__phi[] };
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [N, c] -> { Stmt_bb8[i0] -> MemRef_A[i0] };
; CHECK-NOT: Access
; CHECK:    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %c, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  %tmp1 = sext i32 %c to i64
  br label %bb2

bb2:                                              ; preds = %bb10, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb10 ], [ 0, %bb ]
  %tmp3 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp3, label %bb4, label %bb11

bb4:                                              ; preds = %bb2
  %tmp5 = icmp sgt i64 %indvars.iv, %tmp1
  br i1 %tmp5, label %bb6, label %bb7

bb6:                                              ; preds = %bb4
  br label %bb8

bb7:                                              ; preds = %bb4
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %tmp.0 = phi i32 [ 3, %bb6 ], [ 5, %bb7 ]
  %tmp9 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp.0, ptr %tmp9, align 4
  br label %bb10

bb10:                                             ; preds = %bb8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb2

bb11:                                             ; preds = %bb2
  ret void
}
