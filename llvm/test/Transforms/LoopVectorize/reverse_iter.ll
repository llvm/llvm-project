; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; PR15882: This test ensures that we do not produce wrapping arithmetic when
; creating constant reverse step vectors.
;
; int foo(int n, int *A) {
;   int sum;
;   for (int i=n; i > 0; i--)
;     sum += A[i*2];
;   return sum;
; }
;

;CHECK-LABEL: @foo(
;CHECK:  <i32 0, i32 -1, i32 -2, i32 -3>
;CHECK: ret
define i32 @foo(i32 %n, i32* nocapture %A) {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = sext i32 %n to i64
  br label %3

; <label>:3                                       ; preds = %.lr.ph, %3
  %indvars.iv = phi i64 [ %2, %.lr.ph ], [ %indvars.iv.next, %3 ]
  %sum.01 = phi i32 [ undef, %.lr.ph ], [ %9, %3 ]
  %4 = trunc i64 %indvars.iv to i32
  %5 = shl nsw i32 %4, 1
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds i32, i32* %A, i64 %6
  %8 = load i32, i32* %7, align 4
  %9 = add nsw i32 %8, %sum.01
  %indvars.iv.next = add i64 %indvars.iv, -1
  %10 = trunc i64 %indvars.iv.next to i32
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %3, label %._crit_edge

._crit_edge:                                      ; preds = %3, %0
  %sum.0.lcssa = phi i32 [ undef, %0 ], [ %9, %3 ]
  ret i32 %sum.0.lcssa
}

