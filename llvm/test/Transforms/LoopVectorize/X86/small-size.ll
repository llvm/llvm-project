; RUN: opt < %s -basicaa -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -loop-vectorize-with-block-frequency -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@b = common global [2048 x i32] zeroinitializer, align 16
@c = common global [2048 x i32] zeroinitializer, align 16
@a = common global [2048 x i32] zeroinitializer, align 16
@G = common global [32 x [1024 x i32]] zeroinitializer, align 16
@ub = common global [1024 x i32] zeroinitializer, align 16
@uc = common global [1024 x i32] zeroinitializer, align 16
@d = common global [2048 x i32] zeroinitializer, align 16
@fa = common global [1024 x float] zeroinitializer, align 16
@fb = common global [1024 x float] zeroinitializer, align 16
@ic = common global [1024 x i32] zeroinitializer, align 16
@da = common global [1024 x float] zeroinitializer, align 16
@db = common global [1024 x float] zeroinitializer, align 16
@dc = common global [1024 x float] zeroinitializer, align 16
@dd = common global [1024 x float] zeroinitializer, align 16
@dj = common global [1024 x i32] zeroinitializer, align 16

; We can optimize this test without a tail.
;CHECK-LABEL: @example1(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret void
define void @example1() optsize {
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

; Can't vectorize in 'optsize' mode because we need a tail.
;CHECK-LABEL: @example2(
;CHECK-NOT: store <4 x i32>
;CHECK: ret void
define void @example2(i32 %n, i32 %x) optsize {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph5, label %.preheader

..preheader_crit_edge:                            ; preds = %.lr.ph5
  %phitmp = sext i32 %n to i64
  br label %.preheader

.preheader:                                       ; preds = %..preheader_crit_edge, %0
  %i.0.lcssa = phi i64 [ %phitmp, %..preheader_crit_edge ], [ 0, %0 ]
  %2 = icmp eq i32 %n, 0
  br i1 %2, label %._crit_edge, label %.lr.ph

.lr.ph5:                                          ; preds = %0, %.lr.ph5
  %indvars.iv6 = phi i64 [ %indvars.iv.next7, %.lr.ph5 ], [ 0, %0 ]
  %3 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %indvars.iv6
  store i32 %x, i32* %3, align 4
  %indvars.iv.next7 = add i64 %indvars.iv6, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next7 to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %..preheader_crit_edge, label %.lr.ph5

.lr.ph:                                           ; preds = %.preheader, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ %i.0.lcssa, %.preheader ]
  %.02 = phi i32 [ %4, %.lr.ph ], [ %n, %.preheader ]
  %4 = add nsw i32 %.02, -1
  %5 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %indvars.iv
  %6 = load i32, i32* %5, align 4
  %7 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i64 0, i64 %indvars.iv
  %8 = load i32, i32* %7, align 4
  %9 = and i32 %8, %6
  %10 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %9, i32* %10, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %11 = icmp eq i32 %4, 0
  br i1 %11, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %.preheader
  ret void
}

; N is unknown, we need a tail. Can't vectorize.
;CHECK-LABEL: @example3(
;CHECK-NOT: <4 x i32>
;CHECK: ret void
define void @example3(i32 %n, i32* noalias nocapture %p, i32* noalias nocapture %q) optsize {
  %1 = icmp eq i32 %n, 0
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %0, %.lr.ph
  %.05 = phi i32 [ %2, %.lr.ph ], [ %n, %0 ]
  %.014 = phi i32* [ %5, %.lr.ph ], [ %p, %0 ]
  %.023 = phi i32* [ %3, %.lr.ph ], [ %q, %0 ]
  %2 = add nsw i32 %.05, -1
  %3 = getelementptr inbounds i32, i32* %.023, i64 1
  %4 = load i32, i32* %.023, align 16
  %5 = getelementptr inbounds i32, i32* %.014, i64 1
  store i32 %4, i32* %.014, align 16
  %6 = icmp eq i32 %2, 0
  br i1 %6, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

; We can't vectorize this one because we need a runtime ptr check.
;CHECK-LABEL: @example23(
;CHECK-NOT: <4 x i32>
;CHECK: ret void
define void @example23(i16* nocapture %src, i32* nocapture %dst) optsize {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %.04 = phi i16* [ %src, %0 ], [ %2, %1 ]
  %.013 = phi i32* [ %dst, %0 ], [ %6, %1 ]
  %i.02 = phi i32 [ 0, %0 ], [ %7, %1 ]
  %2 = getelementptr inbounds i16, i16* %.04, i64 1
  %3 = load i16, i16* %.04, align 2
  %4 = zext i16 %3 to i32
  %5 = shl nuw nsw i32 %4, 7
  %6 = getelementptr inbounds i32, i32* %.013, i64 1
  store i32 %5, i32* %.013, align 4
  %7 = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %7, 256
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret void
}


; We CAN vectorize this example because the pointers are marked as noalias.
;CHECK-LABEL: @example23b(
;CHECK: <4 x i32>
;CHECK: ret void
define void @example23b(i16* noalias nocapture %src, i32* noalias nocapture %dst) optsize {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %.04 = phi i16* [ %src, %0 ], [ %2, %1 ]
  %.013 = phi i32* [ %dst, %0 ], [ %6, %1 ]
  %i.02 = phi i32 [ 0, %0 ], [ %7, %1 ]
  %2 = getelementptr inbounds i16, i16* %.04, i64 1
  %3 = load i16, i16* %.04, align 2
  %4 = zext i16 %3 to i32
  %5 = shl nuw nsw i32 %4, 7
  %6 = getelementptr inbounds i32, i32* %.013, i64 1
  store i32 %5, i32* %.013, align 4
  %7 = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %7, 256
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret void
}


