; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(float a[100][100]) {
;      float x;
;
;      for (int i = 0; i < 100; i++) {
;        for (int j = 0; j < 100; j++) {
;          for (int k = 0; k < 100; k++) {
;            if (k == 0)
;              x = 42;
;            a[i][j] += x;
;            x++;
;          }
;        }
;      }
;    }

; The scop we generate for this kernel has a very large number of statements
; and scalar data-dependences due to x being passed along as SSA value or PHI
; node.

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb5
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb5[i0] : 0 <= i0 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb5[i0] -> [i0, 0, 0, 0, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb5[i0] -> MemRef_x_0__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb5[i0] -> MemRef_x_0[] };
; CHECK-NEXT:     Stmt_bb6
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb6[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb6[i0] -> [i0, 1, 0, 0, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb6[i0] -> MemRef_x_0[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb6[i0] -> MemRef_x_1__phi[] };
; CHECK-NEXT:     Stmt_bb7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] : 0 <= i0 <= 99 and 0 <= i1 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> [i0, 2, i1, 0, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_x_1__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_x_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb7[i0, i1] -> MemRef_x_1_lcssa__phi[] };
; CHECK-NEXT:     Stmt_bb8
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb8[i0, i1] : 0 <= i0 <= 99 and 0 <= i1 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb8[i0, i1] -> [i0, 2, i1, 1, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb8[i0, i1] -> MemRef_x_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb8[i0, i1] -> MemRef_x_2__phi[] };
; CHECK-NEXT:     Stmt_bb9
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb9[i0, i1, i2] : 0 <= i0 <= 99 and 0 <= i1 <= 99 and 0 <= i2 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb9[i0, i1, i2] -> [i0, 2, i1, 2, i2, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb9[i0, i1, i2] -> MemRef_x_2__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb9[i0, i1, i2] -> MemRef_x_2[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb9[i0, i1, i2] -> MemRef_x_2_lcssa__phi[] };
; CHECK-NEXT:     Stmt_bb10
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb10[i0, i1, i2] : 0 <= i0 <= 99 and 0 <= i1 <= 99 and 0 <= i2 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb10[i0, i1, i2] -> [i0, 2, i1, 2, i2, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb10[i0, i1, i2] -> MemRef_x_2[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb10[i0, i1, i2] -> MemRef_x_3__phi[] };
; CHECK-NEXT:     Stmt_bb11
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb11[i0, i1, 0] : 0 <= i0 <= 99 and 0 <= i1 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb11[i0, i1, i2] -> [i0, 2, i1, 2, 0, 2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb11[i0, i1, i2] -> MemRef_x_3__phi[] };
; CHECK-NEXT:     Stmt_bb12
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] : 0 <= i0 <= 99 and 0 <= i1 <= 99 and 0 <= i2 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] -> [i0, 2, i1, 2, i2, 3] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] -> MemRef_x_3__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] -> MemRef_a[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] -> MemRef_a[i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb12[i0, i1, i2] -> MemRef_x_3[] };
; CHECK-NEXT:     Stmt_bb16
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb16[i0, i1, i2] : 0 <= i0 <= 99 and 0 <= i1 <= 99 and 0 <= i2 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb16[i0, i1, i2] -> [i0, 2, i1, 2, i2, 4] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb16[i0, i1, i2] -> MemRef_x_2__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb16[i0, i1, i2] -> MemRef_x_3[] };
; CHECK-NEXT:     Stmt_bb19
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb19[i0, i1] : 0 <= i0 <= 99 and 0 <= i1 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb19[i0, i1] -> [i0, 2, i1, 3, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb19[i0, i1] -> MemRef_x_2_lcssa[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb19[i0, i1] -> MemRef_x_2_lcssa__phi[] };
; CHECK-NEXT:     Stmt_bb20
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb20[i0, i1] : 0 <= i0 <= 99 and 0 <= i1 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb20[i0, i1] -> [i0, 2, i1, 4, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb20[i0, i1] -> MemRef_x_2_lcssa[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb20[i0, i1] -> MemRef_x_1__phi[] };
; CHECK-NEXT:     Stmt_bb21
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb21[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb21[i0] -> [i0, 3, 0, 0, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb21[i0] -> MemRef_x_1_lcssa[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb21[i0] -> MemRef_x_1_lcssa__phi[] };
; CHECK-NEXT:     Stmt_bb22
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_bb22[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_bb22[i0] -> [i0, 4, 0, 0, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb22[i0] -> MemRef_x_1_lcssa[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_bb22[i0] -> MemRef_x_0__phi[] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %a) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb22, %bb
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %bb22 ], [ 0, %bb ]
  %x.0 = phi float [ undef, %bb ], [ %x.1.lcssa, %bb22 ]
  %exitcond4 = icmp ne i64 %indvars.iv2, 100
  br i1 %exitcond4, label %bb6, label %bb23

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb20, %bb6
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb20 ], [ 0, %bb6 ]
  %x.1 = phi float [ %x.0, %bb6 ], [ %x.2.lcssa, %bb20 ]
  %exitcond1 = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond1, label %bb8, label %bb21

bb8:                                              ; preds = %bb7
  br label %bb9

bb9:                                              ; preds = %bb16, %bb8
  %x.2 = phi float [ %x.1, %bb8 ], [ %tmp17, %bb16 ]
  %k.0 = phi i32 [ 0, %bb8 ], [ %tmp18, %bb16 ]
  %exitcond = icmp ne i32 %k.0, 100
  br i1 %exitcond, label %bb10, label %bb19

bb10:                                             ; preds = %bb9
  %tmp = icmp eq i32 %k.0, 0
  br i1 %tmp, label %bb11, label %bb12

bb11:                                             ; preds = %bb10
  br label %bb12

bb12:                                             ; preds = %bb11, %bb10
  %x.3 = phi float [ 4.200000e+01, %bb11 ], [ %x.2, %bb10 ]
  %tmp13 = getelementptr inbounds [100 x float], ptr %a, i64 %indvars.iv2, i64 %indvars.iv
  %tmp14 = load float, ptr %tmp13, align 4
  %tmp15 = fadd float %tmp14, %x.3
  store float %tmp15, ptr %tmp13, align 4
  br label %bb16

bb16:                                             ; preds = %bb12
  %tmp17 = fadd float %x.3, 1.000000e+00
  %tmp18 = add nuw nsw i32 %k.0, 1
  br label %bb9

bb19:                                             ; preds = %bb9
  %x.2.lcssa = phi float [ %x.2, %bb9 ]
  br label %bb20

bb20:                                             ; preds = %bb19
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb7

bb21:                                             ; preds = %bb7
  %x.1.lcssa = phi float [ %x.1, %bb7 ]
  br label %bb22

bb22:                                             ; preds = %bb21
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  br label %bb5

bb23:                                             ; preds = %bb5
  ret void
}
