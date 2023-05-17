; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void pos(float *A, long n) {
;      for (long i = 0; i < 100; i++)
;        A[n % 42] += 1;
;    }
;
;
;    void neg(float *A, long n) {
;      for (long i = 0; i < 100; i++)
;        A[n % (-42)] += 1;
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] : (-n + o0) mod 42 = 0 and -41 <= o0 <= 41 and ((n < 0 and o0 <= 0) or (n >= 0 and o0 >= 0)) }
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] : (-n + o0) mod 42 = 0 and -41 <= o0 <= 41 and ((n < 0 and o0 <= 0) or (n >= 0 and o0 >= 0)) }
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] : (-n + o0) mod 42 = 0 and -41 <= o0 <= 41 and ((n > 0 and o0 >= 0) or (n <= 0 and o0 <= 0)) };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb2[i0] -> MemRef_A[o0] : (-n + o0) mod 42 = 0 and -41 <= o0 <= 41 and ((n > 0 and o0 >= 0) or (n <= 0 and o0 <= 0)) };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @pos(ptr %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, 42
  %tmp3 = getelementptr inbounds float, ptr %A, i64 %tmp
  %tmp4 = load float, ptr %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, ptr %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

define void @neg(ptr %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, -42
  %tmp3 = getelementptr inbounds float, ptr %A, i64 %tmp
  %tmp4 = load float, ptr %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, ptr %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
