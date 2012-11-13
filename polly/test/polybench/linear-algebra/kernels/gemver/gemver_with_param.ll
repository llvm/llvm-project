; RUN: opt %loadPolly  %defaultOpts -polly-detect -polly-cloog -analyze  %s | FileCheck %s
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

define void @scop_func(i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb.nph40.preheader, label %return

bb.nph40.preheader:                               ; preds = %entry
  br label %bb.nph40

bb.nph40:                                         ; preds = %bb.nph40.preheader, %bb3
  %i.041 = phi i64 [ %11, %bb3 ], [ 0, %bb.nph40.preheader ]
  %scevgep66 = getelementptr [4000 x double]* @u1, i64 0, i64 %i.041
  %scevgep67 = getelementptr [4000 x double]* @u2, i64 0, i64 %i.041
  %1 = load double* %scevgep66, align 8
  %2 = load double* %scevgep67, align 8
  br label %bb1

bb1:                                              ; preds = %bb1, %bb.nph40
  %j.039 = phi i64 [ 0, %bb.nph40 ], [ %10, %bb1 ]
  %scevgep63 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %i.041, i64 %j.039
  %scevgep62 = getelementptr [4000 x double]* @v2, i64 0, i64 %j.039
  %scevgep61 = getelementptr [4000 x double]* @v1, i64 0, i64 %j.039
  %3 = load double* %scevgep63, align 8
  %4 = load double* %scevgep61, align 8
  %5 = fmul double %1, %4
  %6 = fadd double %3, %5
  %7 = load double* %scevgep62, align 8
  %8 = fmul double %2, %7
  %9 = fadd double %6, %8
  store double %9, double* %scevgep63, align 8
  %10 = add nsw i64 %j.039, 1
  %exitcond16 = icmp eq i64 %10, %n
  br i1 %exitcond16, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %11 = add nsw i64 %i.041, 1
  %exitcond20 = icmp eq i64 %11, %n
  br i1 %exitcond20, label %bb10.preheader, label %bb.nph40

bb10.preheader:                                   ; preds = %bb3
  br i1 %0, label %bb.nph38.bb.nph38.split_crit_edge, label %return

bb.nph30:                                         ; preds = %bb.nph38.bb.nph38.split_crit_edge, %bb9
  %i.134 = phi i64 [ 0, %bb.nph38.bb.nph38.split_crit_edge ], [ %18, %bb9 ]
  %scevgep59 = getelementptr [4000 x double]* @x, i64 0, i64 %i.134
  %.promoted31 = load double* %scevgep59
  br label %bb7

bb7:                                              ; preds = %bb7, %bb.nph30
  %.tmp.032 = phi double [ %.promoted31, %bb.nph30 ], [ %16, %bb7 ]
  %j.129 = phi i64 [ 0, %bb.nph30 ], [ %17, %bb7 ]
  %scevgep56 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %j.129, i64 %i.134
  %scevgep55 = getelementptr [4000 x double]* @y, i64 0, i64 %j.129
  %12 = load double* %scevgep56, align 8
  %13 = fmul double %12, %19
  %14 = load double* %scevgep55, align 8
  %15 = fmul double %13, %14
  %16 = fadd double %.tmp.032, %15
  %17 = add nsw i64 %j.129, 1
  %exitcond10 = icmp eq i64 %17, %n
  br i1 %exitcond10, label %bb9, label %bb7

bb9:                                              ; preds = %bb7
  %.lcssa9 = phi double [ %16, %bb7 ]
  store double %.lcssa9, double* %scevgep59
  %18 = add nsw i64 %i.134, 1
  %exitcond13 = icmp eq i64 %18, %n
  br i1 %exitcond13, label %bb13.preheader, label %bb.nph30

bb.nph38.bb.nph38.split_crit_edge:                ; preds = %bb10.preheader
  %19 = load double* @beta, align 8
  br label %bb.nph30

bb13.preheader:                                   ; preds = %bb9
  br i1 %0, label %bb12.preheader, label %return

bb12.preheader:                                   ; preds = %bb13.preheader
  br label %bb12

bb12:                                             ; preds = %bb12.preheader, %bb12
  %i.227 = phi i64 [ %23, %bb12 ], [ 0, %bb12.preheader ]
  %scevgep52 = getelementptr [4000 x double]* @z, i64 0, i64 %i.227
  %scevgep51 = getelementptr [4000 x double]* @x, i64 0, i64 %i.227
  %20 = load double* %scevgep51, align 8
  %21 = load double* %scevgep52, align 8
  %22 = fadd double %20, %21
  store double %22, double* %scevgep51, align 8
  %23 = add nsw i64 %i.227, 1
  %exitcond6 = icmp eq i64 %23, %n
  br i1 %exitcond6, label %bb19.preheader, label %bb12

bb19.preheader:                                   ; preds = %bb12
  br i1 %0, label %bb.nph26.bb.nph26.split_crit_edge, label %return

bb.nph:                                           ; preds = %bb.nph26.bb.nph26.split_crit_edge, %bb18
  %i.322 = phi i64 [ 0, %bb.nph26.bb.nph26.split_crit_edge ], [ %30, %bb18 ]
  %scevgep49 = getelementptr [4000 x double]* @w, i64 0, i64 %i.322
  %.promoted = load double* %scevgep49
  br label %bb16

bb16:                                             ; preds = %bb16, %bb.nph
  %.tmp.0 = phi double [ %.promoted, %bb.nph ], [ %28, %bb16 ]
  %j.221 = phi i64 [ 0, %bb.nph ], [ %29, %bb16 ]
  %scevgep46 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %i.322, i64 %j.221
  %scevgep = getelementptr [4000 x double]* @x, i64 0, i64 %j.221
  %24 = load double* %scevgep46, align 8
  %25 = fmul double %24, %31
  %26 = load double* %scevgep, align 8
  %27 = fmul double %25, %26
  %28 = fadd double %.tmp.0, %27
  %29 = add nsw i64 %j.221, 1
  %exitcond1 = icmp eq i64 %29, %n
  br i1 %exitcond1, label %bb18, label %bb16

bb18:                                             ; preds = %bb16
  %.lcssa = phi double [ %28, %bb16 ]
  store double %.lcssa, double* %scevgep49
  %30 = add nsw i64 %i.322, 1
  %exitcond = icmp eq i64 %30, %n
  br i1 %exitcond, label %return.loopexit, label %bb.nph

bb.nph26.bb.nph26.split_crit_edge:                ; preds = %bb19.preheader
  %31 = load double* @alpha, align 8
  br label %bb.nph

return.loopexit:                                  ; preds = %bb18
  br label %return

return:                                           ; preds = %return.loopexit, %bb19.preheader, %bb13.preheader, %bb10.preheader, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
