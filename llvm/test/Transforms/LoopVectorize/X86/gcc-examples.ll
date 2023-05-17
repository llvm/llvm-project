; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 -S | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 -force-vector-interleave=0 -S | FileCheck %s -check-prefix=UNROLL

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16
@a = common global [2048 x i32] zeroinitializer, align 16

; Select VF = 8;
;CHECK-LABEL: @example1(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret void

;UNROLL-LABEL: @example1(
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: ret void
define void @example1() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [2048 x i32], ptr @b, i64 0, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds [2048 x i32], ptr @c, i64 0, i64 %indvars.iv
  %5 = load i32, ptr %4, align 4
  %6 = add nsw i32 %5, %3
  %7 = getelementptr inbounds [2048 x i32], ptr @a, i64 0, i64 %indvars.iv
  store i32 %6, ptr %7, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret void
}

; Select VF=4 because sext <8 x i1> to <8 x i32> is expensive.
;CHECK-LABEL: @example10b(
;CHECK: load <4 x i16>
;CHECK: sext <4 x i16>
;CHECK: store <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example10b(
;UNROLL: load <4 x i16>
;UNROLL: load <4 x i16>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: ret void
define void @example10b(ptr noalias nocapture %sa, ptr noalias nocapture %sb, ptr noalias nocapture %sc, ptr noalias nocapture %ia, ptr noalias nocapture %ib, ptr noalias nocapture %ic) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i16, ptr %sb, i64 %indvars.iv
  %3 = load i16, ptr %2, align 2
  %4 = sext i16 %3 to i32
  %5 = getelementptr inbounds i32, ptr %ia, i64 %indvars.iv
  store i32 %4, ptr %5, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %6, label %1

; <label>:6                                       ; preds = %1
  ret void
}

