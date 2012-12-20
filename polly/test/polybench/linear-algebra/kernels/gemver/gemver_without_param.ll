; RUN: opt %loadPolly  %defaultOpts -polly-detect -polly-ast -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@alpha = common global double 0.000000e+00
@beta = common global double 0.000000e+00
@u1 = common global [4000 x double] zeroinitializer, align 32
@u2 = common global [4000 x double] zeroinitializer, align 32
@v1 = common global [4000 x double] zeroinitializer, align 32
@v2 = common global [4000 x double] zeroinitializer, align 32
@y = common global [4000 x double] zeroinitializer, align 32
@z = common global [4000 x double] zeroinitializer, align 32
@x = common global [4000 x double] zeroinitializer, align 32
@w = common global [4000 x double] zeroinitializer, align 32
@A = common global [4000 x [4000 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@B = common global [4000 x [4000 x double]] zeroinitializer, align 32

define void @scop_func() nounwind {
bb.nph31.bb.nph31.split_crit_edge:
  br label %bb.nph26

bb.nph26:                                         ; preds = %bb3, %bb.nph31.bb.nph31.split_crit_edge
  %storemerge27 = phi i64 [ 0, %bb.nph31.bb.nph31.split_crit_edge ], [ %10, %bb3 ]
  %scevgep52 = getelementptr [4000 x double]* @u1, i64 0, i64 %storemerge27
  %scevgep53 = getelementptr [4000 x double]* @u2, i64 0, i64 %storemerge27
  %0 = load double* %scevgep52, align 8
  %1 = load double* %scevgep53, align 8
  br label %bb1

bb1:                                              ; preds = %bb1, %bb.nph26
  %storemerge625 = phi i64 [ 0, %bb.nph26 ], [ %9, %bb1 ]
  %scevgep47 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %storemerge27, i64 %storemerge625
  %scevgep49 = getelementptr [4000 x double]* @v2, i64 0, i64 %storemerge625
  %scevgep48 = getelementptr [4000 x double]* @v1, i64 0, i64 %storemerge625
  %2 = load double* %scevgep47, align 8
  %3 = load double* %scevgep48, align 8
  %4 = fmul double %0, %3
  %5 = fadd double %2, %4
  %6 = load double* %scevgep49, align 8
  %7 = fmul double %1, %6
  %8 = fadd double %5, %7
  store double %8, double* %scevgep47, align 8
  %9 = add nsw i64 %storemerge625, 1
  %exitcond16 = icmp eq i64 %9, 4000
  br i1 %exitcond16, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %10 = add nsw i64 %storemerge27, 1
  %exitcond20 = icmp eq i64 %10, 4000
  br i1 %exitcond20, label %bb.nph24.bb.nph24.split_crit_edge, label %bb.nph26

bb.nph16:                                         ; preds = %bb.nph24.bb.nph24.split_crit_edge, %bb9
  %storemerge120 = phi i64 [ 0, %bb.nph24.bb.nph24.split_crit_edge ], [ %17, %bb9 ]
  %scevgep45 = getelementptr [4000 x double]* @x, i64 0, i64 %storemerge120
  %.promoted17 = load double* %scevgep45
  br label %bb7

bb7:                                              ; preds = %bb7, %bb.nph16
  %.tmp.018 = phi double [ %.promoted17, %bb.nph16 ], [ %15, %bb7 ]
  %storemerge515 = phi i64 [ 0, %bb.nph16 ], [ %16, %bb7 ]
  %scevgep42 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %storemerge515, i64 %storemerge120
  %scevgep41 = getelementptr [4000 x double]* @y, i64 0, i64 %storemerge515
  %11 = load double* %scevgep42, align 8
  %12 = fmul double %11, %18
  %13 = load double* %scevgep41, align 8
  %14 = fmul double %12, %13
  %15 = fadd double %.tmp.018, %14
  %16 = add nsw i64 %storemerge515, 1
  %exitcond10 = icmp eq i64 %16, 4000
  br i1 %exitcond10, label %bb9, label %bb7

bb9:                                              ; preds = %bb7
  %.lcssa9 = phi double [ %15, %bb7 ]
  store double %.lcssa9, double* %scevgep45
  %17 = add nsw i64 %storemerge120, 1
  %exitcond13 = icmp eq i64 %17, 4000
  br i1 %exitcond13, label %bb12.preheader, label %bb.nph16

bb12.preheader:                                   ; preds = %bb9
  br label %bb12

bb.nph24.bb.nph24.split_crit_edge:                ; preds = %bb3
  %18 = load double* @beta, align 8
  br label %bb.nph16

bb12:                                             ; preds = %bb12.preheader, %bb12
  %storemerge213 = phi i64 [ %22, %bb12 ], [ 0, %bb12.preheader ]
  %scevgep38 = getelementptr [4000 x double]* @x, i64 0, i64 %storemerge213
  %scevgep37 = getelementptr [4000 x double]* @z, i64 0, i64 %storemerge213
  %19 = load double* %scevgep38, align 8
  %20 = load double* %scevgep37, align 8
  %21 = fadd double %19, %20
  store double %21, double* %scevgep38, align 8
  %22 = add nsw i64 %storemerge213, 1
  %exitcond6 = icmp eq i64 %22, 4000
  br i1 %exitcond6, label %bb.nph12.bb.nph12.split_crit_edge, label %bb12

bb.nph:                                           ; preds = %bb.nph12.bb.nph12.split_crit_edge, %bb18
  %storemerge38 = phi i64 [ 0, %bb.nph12.bb.nph12.split_crit_edge ], [ %29, %bb18 ]
  %scevgep35 = getelementptr [4000 x double]* @w, i64 0, i64 %storemerge38
  %.promoted = load double* %scevgep35
  br label %bb16

bb16:                                             ; preds = %bb16, %bb.nph
  %.tmp.0 = phi double [ %.promoted, %bb.nph ], [ %27, %bb16 ]
  %storemerge47 = phi i64 [ 0, %bb.nph ], [ %28, %bb16 ]
  %scevgep32 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %storemerge38, i64 %storemerge47
  %scevgep = getelementptr [4000 x double]* @x, i64 0, i64 %storemerge47
  %23 = load double* %scevgep32, align 8
  %24 = fmul double %23, %30
  %25 = load double* %scevgep, align 8
  %26 = fmul double %24, %25
  %27 = fadd double %.tmp.0, %26
  %28 = add nsw i64 %storemerge47, 1
  %exitcond1 = icmp eq i64 %28, 4000
  br i1 %exitcond1, label %bb18, label %bb16

bb18:                                             ; preds = %bb16
  %.lcssa = phi double [ %27, %bb16 ]
  store double %.lcssa, double* %scevgep35
  %29 = add nsw i64 %storemerge38, 1
  %exitcond = icmp eq i64 %29, 4000
  br i1 %exitcond, label %return, label %bb.nph

bb.nph12.bb.nph12.split_crit_edge:                ; preds = %bb12
  %30 = load double* @alpha, align 8
  br label %bb.nph

return:                                           ; preds = %bb18
  ret void
}
; CHECK: for region: 'bb.nph26 => return' in function 'scop_func':
