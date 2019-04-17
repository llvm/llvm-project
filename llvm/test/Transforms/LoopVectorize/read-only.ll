; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @read_only_func(
;CHECK: load <4 x i32>
;CHECK: ret i32
define i32 @read_only_func(i32* nocapture %A, i32* nocapture %B, i32 %n) nounwind uwtable readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i64 %indvars.iv, 13
  %5 = getelementptr inbounds i32, i32* %B, i64 %4
  %6 = load i32, i32* %5, align 4
  %7 = shl i32 %6, 1
  %8 = add i32 %3, %sum.02
  %9 = add i32 %8, %7
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}
