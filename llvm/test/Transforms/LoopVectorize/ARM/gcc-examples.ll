; RUN: opt < %s  -loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -mcpu=swift -S -dce | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16
@a = common global [2048 x i32] zeroinitializer, align 16

; Select VF = 8;
;CHECK-LABEL: @example1(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret void
define void @example1() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i64 0, i64 %indvars.iv
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %5, %3
  %7 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %6, i32* %7, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret void
}

;CHECK-LABEL: @example10b(
;CHECK: load <4 x i16>
;CHECK: sext <4 x i16>
;CHECK: store <4 x i32>
;CHECK: ret void
define void @example10b(i16* noalias nocapture %sa, i16* noalias nocapture %sb, i16* noalias nocapture %sc, i32* noalias nocapture %ia, i32* noalias nocapture %ib, i32* noalias nocapture %ic) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i16, i16* %sb, i64 %indvars.iv
  %3 = load i16, i16* %2, align 2
  %4 = sext i16 %3 to i32
  %5 = getelementptr inbounds i32, i32* %ia, i64 %indvars.iv
  store i32 %4, i32* %5, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %6, label %1

; <label>:6                                       ; preds = %1
  ret void
}

