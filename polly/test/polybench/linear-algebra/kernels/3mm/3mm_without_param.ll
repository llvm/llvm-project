; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [512 x [512 x double]] zeroinitializer, align 32
@B = common global [512 x [512 x double]] zeroinitializer, align 32
@C = common global [512 x [512 x double]] zeroinitializer, align 32
@D = common global [512 x [512 x double]] zeroinitializer, align 32
@E = common global [512 x [512 x double]] zeroinitializer, align 32
@F = common global [512 x [512 x double]] zeroinitializer, align 32
@G = common global [512 x [512 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func() nounwind {
bb.nph76.bb.nph76.split_crit_edge:
  br label %bb5.preheader

bb4.us:                                           ; preds = %bb2.us
  %.lcssa19 = phi double [ %4, %bb2.us ]
  store double %.lcssa19, double* %scevgep94
  %0 = add nsw i64 %storemerge758.us, 1
  %exitcond23 = icmp eq i64 %0, 512
  br i1 %exitcond23, label %bb6, label %bb.nph54.us

bb2.us:                                           ; preds = %bb.nph54.us, %bb2.us
  %.tmp.056.us = phi double [ 0.000000e+00, %bb.nph54.us ], [ %4, %bb2.us ]
  %storemerge853.us = phi i64 [ 0, %bb.nph54.us ], [ %5, %bb2.us ]
  %scevgep91 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %storemerge63, i64 %storemerge853.us
  %scevgep90 = getelementptr [512 x [512 x double]]* @B, i64 0, i64 %storemerge853.us, i64 %storemerge758.us
  %1 = load double* %scevgep91, align 8
  %2 = load double* %scevgep90, align 8
  %3 = fmul double %1, %2
  %4 = fadd double %.tmp.056.us, %3
  %5 = add nsw i64 %storemerge853.us, 1
  %exitcond20 = icmp eq i64 %5, 512
  br i1 %exitcond20, label %bb4.us, label %bb2.us

bb.nph54.us:                                      ; preds = %bb5.preheader, %bb4.us
  %storemerge758.us = phi i64 [ %0, %bb4.us ], [ 0, %bb5.preheader ]
  %scevgep94 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %storemerge63, i64 %storemerge758.us
  store double 0.000000e+00, double* %scevgep94, align 8
  br label %bb2.us

bb6:                                              ; preds = %bb4.us
  %6 = add nsw i64 %storemerge63, 1
  %exitcond26 = icmp ne i64 %6, 512
  br i1 %exitcond26, label %bb5.preheader, label %bb14.preheader.preheader

bb14.preheader.preheader:                         ; preds = %bb6
  br label %bb14.preheader

bb5.preheader:                                    ; preds = %bb6, %bb.nph76.bb.nph76.split_crit_edge
  %storemerge63 = phi i64 [ 0, %bb.nph76.bb.nph76.split_crit_edge ], [ %6, %bb6 ]
  br label %bb.nph54.us

bb13.us:                                          ; preds = %bb11.us
  %.lcssa9 = phi double [ %11, %bb11.us ]
  store double %.lcssa9, double* %scevgep87
  %7 = add nsw i64 %storemerge534.us, 1
  %exitcond13 = icmp eq i64 %7, 512
  br i1 %exitcond13, label %bb15, label %bb.nph30.us

bb11.us:                                          ; preds = %bb.nph30.us, %bb11.us
  %.tmp.032.us = phi double [ 0.000000e+00, %bb.nph30.us ], [ %11, %bb11.us ]
  %storemerge629.us = phi i64 [ 0, %bb.nph30.us ], [ %12, %bb11.us ]
  %scevgep84 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %storemerge139, i64 %storemerge629.us
  %scevgep83 = getelementptr [512 x [512 x double]]* @D, i64 0, i64 %storemerge629.us, i64 %storemerge534.us
  %8 = load double* %scevgep84, align 8
  %9 = load double* %scevgep83, align 8
  %10 = fmul double %8, %9
  %11 = fadd double %.tmp.032.us, %10
  %12 = add nsw i64 %storemerge629.us, 1
  %exitcond10 = icmp eq i64 %12, 512
  br i1 %exitcond10, label %bb13.us, label %bb11.us

bb.nph30.us:                                      ; preds = %bb14.preheader, %bb13.us
  %storemerge534.us = phi i64 [ %7, %bb13.us ], [ 0, %bb14.preheader ]
  %scevgep87 = getelementptr [512 x [512 x double]]* @F, i64 0, i64 %storemerge139, i64 %storemerge534.us
  store double 0.000000e+00, double* %scevgep87, align 8
  br label %bb11.us

bb15:                                             ; preds = %bb13.us
  %13 = add nsw i64 %storemerge139, 1
  %exitcond16 = icmp ne i64 %13, 512
  br i1 %exitcond16, label %bb14.preheader, label %bb23.preheader.preheader

bb23.preheader.preheader:                         ; preds = %bb15
  br label %bb23.preheader

bb14.preheader:                                   ; preds = %bb14.preheader.preheader, %bb15
  %storemerge139 = phi i64 [ %13, %bb15 ], [ 0, %bb14.preheader.preheader ]
  br label %bb.nph30.us

bb22.us:                                          ; preds = %bb20.us
  %.lcssa = phi double [ %18, %bb20.us ]
  store double %.lcssa, double* %scevgep80
  %14 = add nsw i64 %storemerge310.us, 1
  %exitcond = icmp eq i64 %14, 512
  br i1 %exitcond, label %bb24, label %bb.nph.us

bb20.us:                                          ; preds = %bb.nph.us, %bb20.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %18, %bb20.us ]
  %storemerge49.us = phi i64 [ 0, %bb.nph.us ], [ %19, %bb20.us ]
  %scevgep77 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %storemerge215, i64 %storemerge49.us
  %scevgep = getelementptr [512 x [512 x double]]* @F, i64 0, i64 %storemerge49.us, i64 %storemerge310.us
  %15 = load double* %scevgep77, align 8
  %16 = load double* %scevgep, align 8
  %17 = fmul double %15, %16
  %18 = fadd double %.tmp.0.us, %17
  %19 = add nsw i64 %storemerge49.us, 1
  %exitcond1 = icmp eq i64 %19, 512
  br i1 %exitcond1, label %bb22.us, label %bb20.us

bb.nph.us:                                        ; preds = %bb23.preheader, %bb22.us
  %storemerge310.us = phi i64 [ %14, %bb22.us ], [ 0, %bb23.preheader ]
  %scevgep80 = getelementptr [512 x [512 x double]]* @G, i64 0, i64 %storemerge215, i64 %storemerge310.us
  store double 0.000000e+00, double* %scevgep80, align 8
  br label %bb20.us

bb24:                                             ; preds = %bb22.us
  %20 = add nsw i64 %storemerge215, 1
  %exitcond6 = icmp ne i64 %20, 512
  br i1 %exitcond6, label %bb23.preheader, label %return

bb23.preheader:                                   ; preds = %bb23.preheader.preheader, %bb24
  %storemerge215 = phi i64 [ %20, %bb24 ], [ 0, %bb23.preheader.preheader ]
  br label %bb.nph.us

return:                                           ; preds = %bb24
  ret void
}
; CHECK: Valid Region for Scop: bb5.preheader => return
