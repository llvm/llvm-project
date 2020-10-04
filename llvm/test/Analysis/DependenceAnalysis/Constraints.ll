; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1
; RUN: opt < %s -analyze -enable-new-pm=0 -basic-aa -da
;; Check that this code doesn't abort. Test case is reduced version of lnt Polybench benchmark test case dynprog.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@sum_c = common global [10 x [10 x [10 x i32]]] zeroinitializer
@c = common global [10 x [10 x i32]] zeroinitializer
@W = common global [10 x [10 x i32]] zeroinitializer
@out_l = common global i32 0

; Function Attrs: nounwind uwtable
define void @dep_constraint_crash_test(i32 %M, i32 %N) {
  %1 = icmp sgt i32 %N, 0
  br i1 %1, label %.preheader.lr.ph, label %35

.preheader.lr.ph:                                 ; preds = %0
  %2 = add nsw i32 %M, -2
  %3 = icmp slt i32 %M, 2
  %4 = add nsw i32 %M, -1
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 0, i64 %5
  %7 = add nsw i32 %M, -1
  %out_l.promoted = load i32, i32* @out_l
  %8 = sext i32 %7 to i64
  %9 = sext i32 %2 to i64
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge7, %.preheader.lr.ph
  %10 = phi i32 [ %out_l.promoted, %.preheader.lr.ph ], [ %33, %._crit_edge7 ]
  %iter.08 = phi i32 [ 0, %.preheader.lr.ph ], [ %34, %._crit_edge7 ]
  br i1 %3, label %._crit_edge7, label %.lr.ph6

.loopexit:                                        ; preds = %._crit_edge, %.lr.ph6
  %11 = icmp slt i64 %indvars.iv23, %9
  %indvars.iv.next18 = add nuw nsw i64 %indvars.iv17, 1
  %indvars.iv.next14 = add nuw i32 %indvars.iv13, 1
  br i1 %11, label %.lr.ph6, label %._crit_edge7

.lr.ph6:                                          ; preds = %.preheader, %.loopexit
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %.loopexit ], [ 0, %.preheader ]
  %indvars.iv17 = phi i64 [ %indvars.iv.next18, %.loopexit ], [ 1, %.preheader ]
  %indvars.iv13 = phi i32 [ %indvars.iv.next14, %.loopexit ], [ 1, %.preheader ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %12 = icmp slt i64 %indvars.iv23, %8
  br i1 %12, label %.lr.ph4, label %.loopexit

.lr.ph4:                                          ; preds = %.lr.ph6, %._crit_edge
  %indvars.iv19 = phi i64 [ %indvars.iv.next20, %._crit_edge ], [ %indvars.iv17, %.lr.ph6 ]
  %indvars.iv15 = phi i32 [ %indvars.iv.next16, %._crit_edge ], [ %indvars.iv13, %.lr.ph6 ]
  %13 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @sum_c, i64 0, i64 %indvars.iv23, i64 %indvars.iv19, i64 %indvars.iv23
  store i32 0, i32* %13
  %14 = add nsw i64 %indvars.iv19, -1
  %15 = icmp slt i64 %indvars.iv23, %14
  br i1 %15, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph4, %.lr.ph
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %.lr.ph ], [ %indvars.iv17, %.lr.ph4 ]
  %16 = add nsw i64 %indvars.iv11, -1
  %17 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @sum_c, i64 0, i64 %indvars.iv23, i64 %indvars.iv19, i64 %16
  %18 = load i32, i32* %17
  %19 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %indvars.iv23, i64 %indvars.iv11
  %20 = load i32, i32* %19
  %21 = add nsw i32 %20, %18
  %22 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %indvars.iv11, i64 %indvars.iv19
  %23 = load i32, i32* %22
  %24 = add nsw i32 %21, %23
  %25 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @sum_c, i64 0, i64 %indvars.iv23, i64 %indvars.iv19, i64 %indvars.iv11
  store i32 %24, i32* %25
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next12 to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %indvars.iv15
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %.lr.ph4
  %26 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* @sum_c, i64 0, i64 %indvars.iv23, i64 %indvars.iv19, i64 %14
  %27 = load i32, i32* %26
  %28 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @W, i64 0, i64 %indvars.iv23, i64 %indvars.iv19
  %29 = load i32, i32* %28
  %30 = add nsw i32 %29, %27
  %31 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %indvars.iv23, i64 %indvars.iv19
  store i32 %30, i32* %31
  %indvars.iv.next16 = add nuw i32 %indvars.iv15, 1
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %lftr.wideiv21 = trunc i64 %indvars.iv.next20 to i32
  %exitcond22 = icmp eq i32 %lftr.wideiv21, %M
  br i1 %exitcond22, label %.loopexit, label %.lr.ph4

._crit_edge7:                                     ; preds = %.loopexit, %.preheader
  %32 = load i32, i32* %6
  %33 = add nsw i32 %10, %32
  %34 = add nuw nsw i32 %iter.08, 1
  %exitcond25 = icmp eq i32 %34, %N
  br i1 %exitcond25, label %._crit_edge9, label %.preheader

._crit_edge9:                                     ; preds = %._crit_edge7
  store i32 %33, i32* @out_l
  br label %35

; <label>:35                                      ; preds = %._crit_edge9, %0
  ret void
}

