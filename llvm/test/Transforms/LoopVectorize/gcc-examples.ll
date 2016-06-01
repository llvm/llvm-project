; RUN: opt < %s  -basicaa -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -dce -instcombine -S | FileCheck %s
; RUN: opt < %s  -basicaa -loop-vectorize -force-vector-width=4 -force-vector-interleave=4 -dce -instcombine -S | FileCheck %s -check-prefix=UNROLL

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

;CHECK-LABEL: @example1(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example1(
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: add nsw <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: ret void
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

;CHECK-LABEL: @example2(
;CHECK: store <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example2(
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: ret void
define void @example2(i32 %n, i32 %x) nounwind uwtable ssp {
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

;CHECK-LABEL: @example3(
;CHECK: <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example3(
;UNROLL: <4 x i32>
;UNROLL: <4 x i32>
;UNROLL: <4 x i32>
;UNROLL: <4 x i32>
;UNROLL: ret void
define void @example3(i32 %n, i32* noalias nocapture %p, i32* noalias nocapture %q) nounwind uwtable ssp {
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

;CHECK-LABEL: @example4(
;CHECK: load <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example4(
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: load <4 x i32>
;UNROLL: ret void
define void @example4(i32 %n, i32* noalias nocapture %p, i32* noalias nocapture %q) nounwind uwtable ssp {
  %1 = add nsw i32 %n, -1
  %2 = icmp eq i32 %n, 0
  br i1 %2, label %.preheader4, label %.lr.ph10

.preheader4:                                      ; preds = %0
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %.lr.ph6, label %._crit_edge

.lr.ph10:                                         ; preds = %0, %.lr.ph10
  %4 = phi i32 [ %9, %.lr.ph10 ], [ %1, %0 ]
  %.018 = phi i32* [ %8, %.lr.ph10 ], [ %p, %0 ]
  %.027 = phi i32* [ %5, %.lr.ph10 ], [ %q, %0 ]
  %5 = getelementptr inbounds i32, i32* %.027, i64 1
  %6 = load i32, i32* %.027, align 16
  %7 = add nsw i32 %6, 5
  %8 = getelementptr inbounds i32, i32* %.018, i64 1
  store i32 %7, i32* %.018, align 16
  %9 = add nsw i32 %4, -1
  %10 = icmp eq i32 %4, 0
  br i1 %10, label %._crit_edge, label %.lr.ph10

.preheader:                                       ; preds = %.lr.ph6
  br i1 %3, label %.lr.ph, label %._crit_edge

.lr.ph6:                                          ; preds = %.preheader4, %.lr.ph6
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %.lr.ph6 ], [ 0, %.preheader4 ]
  %indvars.iv.next12 = add i64 %indvars.iv11, 1
  %11 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %indvars.iv.next12
  %12 = load i32, i32* %11, align 4
  %13 = add nsw i64 %indvars.iv11, 3
  %14 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i64 0, i64 %13
  %15 = load i32, i32* %14, align 4
  %16 = add nsw i32 %15, %12
  %17 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv11
  store i32 %16, i32* %17, align 4
  %lftr.wideiv13 = trunc i64 %indvars.iv.next12 to i32
  %exitcond14 = icmp eq i32 %lftr.wideiv13, %1
  br i1 %exitcond14, label %.preheader, label %.lr.ph6

.lr.ph:                                           ; preds = %.preheader, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.preheader ]
  %18 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  %19 = load i32, i32* %18, align 4
  %20 = icmp sgt i32 %19, 4
  %21 = select i1 %20, i32 4, i32 0
  %22 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %indvars.iv
  store i32 %21, i32* %22, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %1
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph10, %.preheader4, %.lr.ph, %.preheader
  ret void
}

;CHECK-LABEL: @example8(
;CHECK: store <4 x i32>
;CHECK: ret void
;UNROLL-LABEL: @example8(
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: store <4 x i32>
;UNROLL: ret void
define void @example8(i32 %x) nounwind uwtable ssp {
  br label %.preheader

.preheader:                                       ; preds = %3, %0
  %indvars.iv3 = phi i64 [ 0, %0 ], [ %indvars.iv.next4, %3 ]
  br label %1

; <label>:1                                       ; preds = %1, %.preheader
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [32 x [1024 x i32]], [32 x [1024 x i32]]* @G, i64 0, i64 %indvars.iv3, i64 %indvars.iv
  store i32 %x, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %3, label %1

; <label>:3                                       ; preds = %1
  %indvars.iv.next4 = add i64 %indvars.iv3, 1
  %lftr.wideiv5 = trunc i64 %indvars.iv.next4 to i32
  %exitcond6 = icmp eq i32 %lftr.wideiv5, 32
  br i1 %exitcond6, label %4, label %.preheader

; <label>:4                                       ; preds = %3
  ret void
}

;CHECK-LABEL: @example9(
;CHECK: phi <4 x i32>
;CHECK: ret i32
define i32 @example9() nounwind uwtable readonly ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %diff.01 = phi i32 [ 0, %0 ], [ %7, %1 ]
  %2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @ub, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds [1024 x i32], [1024 x i32]* @uc, i64 0, i64 %indvars.iv
  %5 = load i32, i32* %4, align 4
  %6 = add i32 %3, %diff.01
  %7 = sub i32 %6, %5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret i32 %7
}

;CHECK-LABEL: @example10a(
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: load <4 x i16>
;CHECK: add <4 x i16>
;CHECK: store <4 x i16>
;CHECK: ret void
define void @example10a(i16* noalias nocapture %sa, i16* noalias nocapture %sb, i16* noalias nocapture %sc, i32* noalias nocapture %ia, i32* noalias nocapture %ib, i32* noalias nocapture %ic) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i32, i32* %ib, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = getelementptr inbounds i32, i32* %ic, i64 %indvars.iv
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %5, %3
  %7 = getelementptr inbounds i32, i32* %ia, i64 %indvars.iv
  store i32 %6, i32* %7, align 4
  %8 = getelementptr inbounds i16, i16* %sb, i64 %indvars.iv
  %9 = load i16, i16* %8, align 2
  %10 = getelementptr inbounds i16, i16* %sc, i64 %indvars.iv
  %11 = load i16, i16* %10, align 2
  %12 = add i16 %11, %9
  %13 = getelementptr inbounds i16, i16* %sa, i64 %indvars.iv
  store i16 %12, i16* %13, align 2
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %14, label %1

; <label>:14                                      ; preds = %1
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

;CHECK-LABEL: @example11(
;CHECK: load i32
;CHECK: load i32
;CHECK: load i32
;CHECK: load i32
;CHECK: insertelement
;CHECK: insertelement
;CHECK: insertelement
;CHECK: insertelement
;CHECK: ret void
define void @example11() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = shl nsw i64 %indvars.iv, 1
  %3 = or i64 %2, 1
  %4 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %3
  %5 = load i32, i32* %4, align 4
  %6 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i64 0, i64 %3
  %7 = load i32, i32* %6, align 4
  %8 = mul nsw i32 %7, %5
  %9 = getelementptr inbounds [2048 x i32], [2048 x i32]* @b, i64 0, i64 %2
  %10 = load i32, i32* %9, align 8
  %11 = getelementptr inbounds [2048 x i32], [2048 x i32]* @c, i64 0, i64 %2
  %12 = load i32, i32* %11, align 8
  %13 = mul nsw i32 %12, %10
  %14 = sub nsw i32 %8, %13
  %15 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  store i32 %14, i32* %15, align 4
  %16 = mul nsw i32 %7, %10
  %17 = mul nsw i32 %12, %5
  %18 = add nsw i32 %17, %16
  %19 = getelementptr inbounds [2048 x i32], [2048 x i32]* @d, i64 0, i64 %indvars.iv
  store i32 %18, i32* %19, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 512
  br i1 %exitcond, label %20, label %1

; <label>:20                                      ; preds = %1
  ret void
}

;CHECK-LABEL: @example12(
;CHECK: %vec.ind1 = phi <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret void
define void @example12() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  %3 = trunc i64 %indvars.iv to i32
  store i32 %3, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %4, label %1

; <label>:4                                       ; preds = %1
  ret void
}

;CHECK-LABEL: @example13(
;CHECK: <4 x i32>
;CHECK: ret void
define void @example13(i32** nocapture %A, i32** nocapture %B, i32* nocapture %out) nounwind uwtable ssp {
  br label %.preheader

.preheader:                                       ; preds = %14, %0
  %indvars.iv4 = phi i64 [ 0, %0 ], [ %indvars.iv.next5, %14 ]
  %1 = getelementptr inbounds i32*, i32** %A, i64 %indvars.iv4
  %2 = load i32*, i32** %1, align 8
  %3 = getelementptr inbounds i32*, i32** %B, i64 %indvars.iv4
  %4 = load i32*, i32** %3, align 8
  br label %5

; <label>:5                                       ; preds = %.preheader, %5
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %5 ]
  %diff.02 = phi i32 [ 0, %.preheader ], [ %11, %5 ]
  %6 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv
  %7 = load i32, i32* %6, align 4
  %8 = getelementptr inbounds i32, i32* %4, i64 %indvars.iv
  %9 = load i32, i32* %8, align 4
  %10 = add i32 %7, %diff.02
  %11 = sub i32 %10, %9
  %indvars.iv.next = add i64 %indvars.iv, 8
  %12 = trunc i64 %indvars.iv.next to i32
  %13 = icmp slt i32 %12, 1024
  br i1 %13, label %5, label %14

; <label>:14                                      ; preds = %5
  %15 = getelementptr inbounds i32, i32* %out, i64 %indvars.iv4
  store i32 %11, i32* %15, align 4
  %indvars.iv.next5 = add i64 %indvars.iv4, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next5 to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 32
  br i1 %exitcond, label %16, label %.preheader

; <label>:16                                      ; preds = %14
  ret void
}

; Can vectorize.
;CHECK-LABEL: @example14(
;CHECK: <4 x i32>
;CHECK: ret void
define void @example14(i32** nocapture %in, i32** nocapture %coeff, i32* nocapture %out) nounwind uwtable ssp {
.preheader3:
  br label %.preheader

.preheader:                                       ; preds = %11, %.preheader3
  %indvars.iv7 = phi i64 [ 0, %.preheader3 ], [ %indvars.iv.next8, %11 ]
  %sum.05 = phi i32 [ 0, %.preheader3 ], [ %10, %11 ]
  br label %0

; <label>:0                                       ; preds = %0, %.preheader
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %0 ]
  %sum.12 = phi i32 [ %sum.05, %.preheader ], [ %10, %0 ]
  %1 = getelementptr inbounds i32*, i32** %in, i64 %indvars.iv
  %2 = load i32*, i32** %1, align 8
  %3 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv7
  %4 = load i32, i32* %3, align 4
  %5 = getelementptr inbounds i32*, i32** %coeff, i64 %indvars.iv
  %6 = load i32*, i32** %5, align 8
  %7 = getelementptr inbounds i32, i32* %6, i64 %indvars.iv7
  %8 = load i32, i32* %7, align 4
  %9 = mul nsw i32 %8, %4
  %10 = add nsw i32 %9, %sum.12
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %11, label %0

; <label>:11                                      ; preds = %0
  %indvars.iv.next8 = add i64 %indvars.iv7, 1
  %lftr.wideiv9 = trunc i64 %indvars.iv.next8 to i32
  %exitcond10 = icmp eq i32 %lftr.wideiv9, 32
  br i1 %exitcond10, label %.preheader3.1, label %.preheader

.preheader3.1:                                    ; preds = %11
  store i32 %10, i32* %out, align 4
  br label %.preheader.1

.preheader.1:                                     ; preds = %24, %.preheader3.1
  %indvars.iv7.1 = phi i64 [ 0, %.preheader3.1 ], [ %indvars.iv.next8.1, %24 ]
  %sum.05.1 = phi i32 [ 0, %.preheader3.1 ], [ %23, %24 ]
  br label %12

; <label>:12                                      ; preds = %12, %.preheader.1
  %indvars.iv.1 = phi i64 [ 0, %.preheader.1 ], [ %13, %12 ]
  %sum.12.1 = phi i32 [ %sum.05.1, %.preheader.1 ], [ %23, %12 ]
  %13 = add nsw i64 %indvars.iv.1, 1
  %14 = getelementptr inbounds i32*, i32** %in, i64 %13
  %15 = load i32*, i32** %14, align 8
  %16 = getelementptr inbounds i32, i32* %15, i64 %indvars.iv7.1
  %17 = load i32, i32* %16, align 4
  %18 = getelementptr inbounds i32*, i32** %coeff, i64 %indvars.iv.1
  %19 = load i32*, i32** %18, align 8
  %20 = getelementptr inbounds i32, i32* %19, i64 %indvars.iv7.1
  %21 = load i32, i32* %20, align 4
  %22 = mul nsw i32 %21, %17
  %23 = add nsw i32 %22, %sum.12.1
  %lftr.wideiv.1 = trunc i64 %13 to i32
  %exitcond.1 = icmp eq i32 %lftr.wideiv.1, 1024
  br i1 %exitcond.1, label %24, label %12

; <label>:24                                      ; preds = %12
  %indvars.iv.next8.1 = add i64 %indvars.iv7.1, 1
  %lftr.wideiv9.1 = trunc i64 %indvars.iv.next8.1 to i32
  %exitcond10.1 = icmp eq i32 %lftr.wideiv9.1, 32
  br i1 %exitcond10.1, label %.preheader3.2, label %.preheader.1

.preheader3.2:                                    ; preds = %24
  %25 = getelementptr inbounds i32, i32* %out, i64 1
  store i32 %23, i32* %25, align 4
  br label %.preheader.2

.preheader.2:                                     ; preds = %38, %.preheader3.2
  %indvars.iv7.2 = phi i64 [ 0, %.preheader3.2 ], [ %indvars.iv.next8.2, %38 ]
  %sum.05.2 = phi i32 [ 0, %.preheader3.2 ], [ %37, %38 ]
  br label %26

; <label>:26                                      ; preds = %26, %.preheader.2
  %indvars.iv.2 = phi i64 [ 0, %.preheader.2 ], [ %indvars.iv.next.2, %26 ]
  %sum.12.2 = phi i32 [ %sum.05.2, %.preheader.2 ], [ %37, %26 ]
  %27 = add nsw i64 %indvars.iv.2, 2
  %28 = getelementptr inbounds i32*, i32** %in, i64 %27
  %29 = load i32*, i32** %28, align 8
  %30 = getelementptr inbounds i32, i32* %29, i64 %indvars.iv7.2
  %31 = load i32, i32* %30, align 4
  %32 = getelementptr inbounds i32*, i32** %coeff, i64 %indvars.iv.2
  %33 = load i32*, i32** %32, align 8
  %34 = getelementptr inbounds i32, i32* %33, i64 %indvars.iv7.2
  %35 = load i32, i32* %34, align 4
  %36 = mul nsw i32 %35, %31
  %37 = add nsw i32 %36, %sum.12.2
  %indvars.iv.next.2 = add i64 %indvars.iv.2, 1
  %lftr.wideiv.2 = trunc i64 %indvars.iv.next.2 to i32
  %exitcond.2 = icmp eq i32 %lftr.wideiv.2, 1024
  br i1 %exitcond.2, label %38, label %26

; <label>:38                                      ; preds = %26
  %indvars.iv.next8.2 = add i64 %indvars.iv7.2, 1
  %lftr.wideiv9.2 = trunc i64 %indvars.iv.next8.2 to i32
  %exitcond10.2 = icmp eq i32 %lftr.wideiv9.2, 32
  br i1 %exitcond10.2, label %.preheader3.3, label %.preheader.2

.preheader3.3:                                    ; preds = %38
  %39 = getelementptr inbounds i32, i32* %out, i64 2
  store i32 %37, i32* %39, align 4
  br label %.preheader.3

.preheader.3:                                     ; preds = %52, %.preheader3.3
  %indvars.iv7.3 = phi i64 [ 0, %.preheader3.3 ], [ %indvars.iv.next8.3, %52 ]
  %sum.05.3 = phi i32 [ 0, %.preheader3.3 ], [ %51, %52 ]
  br label %40

; <label>:40                                      ; preds = %40, %.preheader.3
  %indvars.iv.3 = phi i64 [ 0, %.preheader.3 ], [ %indvars.iv.next.3, %40 ]
  %sum.12.3 = phi i32 [ %sum.05.3, %.preheader.3 ], [ %51, %40 ]
  %41 = add nsw i64 %indvars.iv.3, 3
  %42 = getelementptr inbounds i32*, i32** %in, i64 %41
  %43 = load i32*, i32** %42, align 8
  %44 = getelementptr inbounds i32, i32* %43, i64 %indvars.iv7.3
  %45 = load i32, i32* %44, align 4
  %46 = getelementptr inbounds i32*, i32** %coeff, i64 %indvars.iv.3
  %47 = load i32*, i32** %46, align 8
  %48 = getelementptr inbounds i32, i32* %47, i64 %indvars.iv7.3
  %49 = load i32, i32* %48, align 4
  %50 = mul nsw i32 %49, %45
  %51 = add nsw i32 %50, %sum.12.3
  %indvars.iv.next.3 = add i64 %indvars.iv.3, 1
  %lftr.wideiv.3 = trunc i64 %indvars.iv.next.3 to i32
  %exitcond.3 = icmp eq i32 %lftr.wideiv.3, 1024
  br i1 %exitcond.3, label %52, label %40

; <label>:52                                      ; preds = %40
  %indvars.iv.next8.3 = add i64 %indvars.iv7.3, 1
  %lftr.wideiv9.3 = trunc i64 %indvars.iv.next8.3 to i32
  %exitcond10.3 = icmp eq i32 %lftr.wideiv9.3, 32
  br i1 %exitcond10.3, label %53, label %.preheader.3

; <label>:53                                      ; preds = %52
  %54 = getelementptr inbounds i32, i32* %out, i64 3
  store i32 %51, i32* %54, align 4
  ret void
}

;CHECK-LABEL: @example21(
;CHECK: load <4 x i32>
;CHECK: shufflevector {{.*}} <i32 3, i32 2, i32 1, i32 0>
;CHECK: ret i32
define i32 @example21(i32* nocapture %b, i32 %n) nounwind uwtable readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = sext i32 %n to i64
  br label %3

; <label>:3                                       ; preds = %.lr.ph, %3
  %indvars.iv = phi i64 [ %2, %.lr.ph ], [ %indvars.iv.next, %3 ]
  %a.02 = phi i32 [ 0, %.lr.ph ], [ %6, %3 ]
  %indvars.iv.next = add i64 %indvars.iv, -1
  %4 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv.next
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %5, %a.02
  %7 = trunc i64 %indvars.iv.next to i32
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %3, label %._crit_edge

._crit_edge:                                      ; preds = %3, %0
  %a.0.lcssa = phi i32 [ 0, %0 ], [ %6, %3 ]
  ret i32 %a.0.lcssa
}

;CHECK-LABEL: @example23(
;CHECK: <4 x i32>
;CHECK: ret void
define void @example23(i16* nocapture %src, i32* nocapture %dst) nounwind uwtable ssp {
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

;CHECK-LABEL: @example24(
;CHECK: shufflevector <4 x i16>
;CHECK: ret void
define void @example24(i16 signext %x, i16 signext %y) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [1024 x float], [1024 x float]* @fa, i64 0, i64 %indvars.iv
  %3 = load float, float* %2, align 4
  %4 = getelementptr inbounds [1024 x float], [1024 x float]* @fb, i64 0, i64 %indvars.iv
  %5 = load float, float* %4, align 4
  %6 = fcmp olt float %3, %5
  %x.y = select i1 %6, i16 %x, i16 %y
  %7 = sext i16 %x.y to i32
  %8 = getelementptr inbounds [1024 x i32], [1024 x i32]* @ic, i64 0, i64 %indvars.iv
  store i32 %7, i32* %8, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %9, label %1

; <label>:9                                       ; preds = %1
  ret void
}

;CHECK-LABEL: @example25(
;CHECK: and <4 x i1>
;CHECK: zext <4 x i1>
;CHECK: ret void
define void @example25() nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [1024 x float], [1024 x float]* @da, i64 0, i64 %indvars.iv
  %3 = load float, float* %2, align 4
  %4 = getelementptr inbounds [1024 x float], [1024 x float]* @db, i64 0, i64 %indvars.iv
  %5 = load float, float* %4, align 4
  %6 = fcmp olt float %3, %5
  %7 = getelementptr inbounds [1024 x float], [1024 x float]* @dc, i64 0, i64 %indvars.iv
  %8 = load float, float* %7, align 4
  %9 = getelementptr inbounds [1024 x float], [1024 x float]* @dd, i64 0, i64 %indvars.iv
  %10 = load float, float* %9, align 4
  %11 = fcmp olt float %8, %10
  %12 = and i1 %6, %11
  %13 = zext i1 %12 to i32
  %14 = getelementptr inbounds [1024 x i32], [1024 x i32]* @dj, i64 0, i64 %indvars.iv
  store i32 %13, i32* %14, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %15, label %1

; <label>:15                                      ; preds = %1
  ret void
}

