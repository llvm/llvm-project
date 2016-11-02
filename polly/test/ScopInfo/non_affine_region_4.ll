; RUN: opt %loadPolly -polly-scops  -analyze < %s | FileCheck %s
;
; Verify that both scalars (x and y) are properly written in the non-affine
; region and read afterwards.
;
;    void f(int *A, int b) {
;      for (int i = 0; i < 1024; i++) {
;        int x = 0, y = 0;
;        if ((x = 1 + A[i]))
;          y++;
;        A[i] = x + y;
;      }
;    }

; CHECK-LABEL: Region: %bb1---%bb11
;
; CHECK:       Arrays {
; CHECK-NEXT:      i32 MemRef_A[*]; // Element size 4
; CHECK-NEXT:      i32 MemRef_y__phi; // Element size 4
; CHECK-NEXT:      i32 MemRef_x; // Element size 4
; CHECK-NEXT:  }
;
; CHECK:       Arrays (Bounds as pw_affs) {
; CHECK-NEXT:      i32 MemRef_A[*]; // Element size 4
; CHECK-NEXT:      i32 MemRef_y__phi; // Element size 4
; CHECK-NEXT:      i32 MemRef_x; // Element size 4
; CHECK-NEXT:  }
;
; CHECK:       Statements {
; CHECK-NEXT:      Stmt_bb2__TO__bb7
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              { Stmt_bb2__TO__bb7[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              { Stmt_bb2__TO__bb7[i0] -> [i0, 0] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              { Stmt_bb2__TO__bb7[i0] -> MemRef_A[i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_bb2__TO__bb7[i0] -> MemRef_y__phi[] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_bb2__TO__bb7[i0] -> MemRef_x[] };
; CHECK-NEXT:      Stmt_bb7
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              { Stmt_bb7[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              { Stmt_bb7[i0] -> [i0, 1] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_bb7[i0] -> MemRef_y__phi[] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_bb7[i0] -> MemRef_x[] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              { Stmt_bb7[i0] -> MemRef_A[i0] };
; CHECK-NEXT:  }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb10, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb10 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb11

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %x = load i32,  i32* %tmp, align 4
  %tmp4 = add nsw i32 %x, 1
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb7, label %bb6

bb6:                                              ; preds = %bb2
  br label %bb7

bb7:                                              ; preds = %bb2, %bb6
  %y = phi i32 [ 1, %bb6 ], [ 0, %bb2 ]
  %tmp4copy = add nsw i32 %x, 1
  %tmp8 = add nsw i32 %tmp4copy, %y
  %tmp9 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp8, i32* %tmp9, align 4
  br label %bb10

bb10:                                             ; preds = %bb7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb11:                                             ; preds = %bb1
  ret void
}
