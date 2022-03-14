; RUN: opt %loadPolly -polly-allow-nonaffine -polly-print-scops -disable-output < %s | FileCheck %s
;
; Verify only the incoming scalar x is modeled as a read in the non-affine
; region.
;
;    void f(int *A, int b) {
;      int x;
;      for (int i = 0; i < 1024; i++) {
;        if (b > i)
;          x = 0;
;        else if (b < 2 * i)
;          x = 3;
;        else
;          x = b;
;
;        if (A[x])
;          A[x] = 0;
;      }
;    }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb3
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [b] -> { Stmt_bb3[i0] : 0 <= i0 <= 1023 and i0 < b };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [b] -> { Stmt_bb3[i0] -> [i0, 2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [b] -> { Stmt_bb3[i0] -> MemRef_x_1__phi[] };
; CHECK-NEXT:     Stmt_bb7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [b] -> { Stmt_bb7[i0] : i0 >= b and 0 <= i0 <= 1023 and 2i0 > b };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [b] -> { Stmt_bb7[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [b] -> { Stmt_bb7[i0] -> MemRef_x_1__phi[] };
; CHECK-NEXT:     Stmt_bb8
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [b] -> { Stmt_bb8[0] : b = 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [b] -> { Stmt_bb8[i0] -> [0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [b] -> { Stmt_bb8[i0] -> MemRef_x_1__phi[] };
; CHECK-NEXT:     Stmt_bb10__TO__bb18
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [b] -> { Stmt_bb10__TO__bb18[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [b] -> { Stmt_bb10__TO__bb18[i0] -> [i0, 3] }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [b] -> { Stmt_bb10__TO__bb18[i0] -> MemRef_x_1__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [b] -> { Stmt_bb10__TO__bb18[i0] -> MemRef_A[o0] };
; CHECK-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [b] -> { Stmt_bb10__TO__bb18[i0] -> MemRef_A[o0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb19, %bb
  %i.0 = phi i32 [ 0, %bb ], [ %tmp20, %bb19 ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb21

bb2:                                              ; preds = %bb1
  %tmp = icmp slt i32 %i.0, %b
  br i1 %tmp, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  br label %bb10

bb4:                                              ; preds = %bb2
  %tmp5 = mul nsw i32 %i.0, 2
  %tmp6 = icmp sgt i32 %tmp5, %b
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb4
  br label %bb10

bb8:                                              ; preds = %bb4
  br label %bb10

bb10:                                             ; preds = %bb9, %bb3
  %x.1 = phi i32 [ 0, %bb3 ], [ 3, %bb7 ], [ %b, %bb8 ]
  %tmp11 = sext i32 %x.1 to i64
  %tmp12 = getelementptr inbounds i32, i32* %A, i64 %tmp11
  %tmp13 = load i32,  i32* %tmp12, align 4
  %tmp14 = icmp eq i32 %tmp13, 0
  br i1 %tmp14, label %bb18, label %bb15

bb15:                                             ; preds = %bb10
  %tmp16 = sext i32 %x.1 to i64
  %tmp17 = getelementptr inbounds i32, i32* %A, i64 %tmp16
  store i32 0, i32* %tmp17, align 4
  br label %bb18

bb18:                                             ; preds = %bb10, %bb15
  br label %bb19

bb19:                                             ; preds = %bb18
  %tmp20 = add nuw nsw i32 %i.0, 1
  br label %bb1

bb21:                                             ; preds = %bb1
  ret void
}
