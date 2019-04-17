; RUN: opt < %s  -loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

;CHECK:foo_F32
;CHECK: <4 x float>
;CHECK:ret
define float @foo_F32(float* nocapture %A, i32 %n) nounwind uwtable readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %prod.01 = phi float [ %4, %.lr.ph ], [ 0.000000e+00, %0 ]
  %2 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %3 = load float, float* %2, align 8
  %4 = fmul fast float %prod.01, %3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %prod.0.lcssa = phi float [ 0.000000e+00, %0 ], [ %4, %.lr.ph ]
  ret float %prod.0.lcssa
}

;CHECK:foo_I8
;CHECK: xor <16 x i8>
;CHECK:ret
define signext i8 @foo_I8(i8* nocapture %A, i32 %n) nounwind uwtable readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %red.01 = phi i8 [ %4, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv
  %3 = load i8, i8* %2, align 1
  %4 = xor i8 %3, %red.01
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %red.0.lcssa = phi i8 [ 0, %0 ], [ %4, %.lr.ph ]
  ret i8 %red.0.lcssa
}


