; RUN: opt < %s -loop-vectorize -force-vector-width=4 -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@array = common global [1024 x i32] zeroinitializer, align 16

;CHECK: @array_at_plus_one
;CHECK: add <4 x i64>
;CHECK: trunc <4 x i64>
;CHECK: add i64 %index, 12
;CHECK: ret i32
define i32 @array_at_plus_one(i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = add nsw i64 %indvars.iv, 12
  %3 = getelementptr inbounds [1024 x i32]* @array, i64 0, i64 %2
  %4 = trunc i64 %indvars.iv to i32
  store i32 %4, i32* %3, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}
