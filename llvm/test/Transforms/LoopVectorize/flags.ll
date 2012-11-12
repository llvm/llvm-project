; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -dce -instcombine -licm -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: @flags1
;CHECK: load <4 x i32>
;CHECK: mul nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret i32
define i32 @flags1(i32 %n, i32* nocapture %A) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 9
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 9, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = mul nsw i32 %3, 3
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}


;CHECK: @flags2
;CHECK: load <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret i32
define i32 @flags2(i32 %n, i32* nocapture %A) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 9
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 9, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = mul i32 %3, 3
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}
