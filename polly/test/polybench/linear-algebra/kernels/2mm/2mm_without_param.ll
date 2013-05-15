; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@alpha1 = common global double 0.000000e+00
@beta1 = common global double 0.000000e+00
@alpha2 = common global double 0.000000e+00
@beta2 = common global double 0.000000e+00
@A = common global [512 x [512 x double]] zeroinitializer, align 32
@B = common global [512 x [512 x double]] zeroinitializer, align 32
@C = common global [512 x [512 x double]] zeroinitializer, align 32
@D = common global [512 x [512 x double]] zeroinitializer, align 32
@E = common global [512 x [512 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func() nounwind {
bb.nph50.bb.nph50.split_crit_edge:
  br label %bb5.preheader

bb4.us:                                           ; preds = %bb2.us
  %.lcssa9 = phi double [ %4, %bb2.us ]
  store double %.lcssa9, double* %scevgep61
  %0 = add nsw i64 %storemerge431.us, 1
  %exitcond13 = icmp eq i64 %0, 512
  br i1 %exitcond13, label %bb6, label %bb.nph27.us

bb2.us:                                           ; preds = %bb.nph27.us, %bb2.us
  %.tmp.029.us = phi double [ 0.000000e+00, %bb.nph27.us ], [ %4, %bb2.us ]
  %storemerge526.us = phi i64 [ 0, %bb.nph27.us ], [ %5, %bb2.us ]
  %scevgep58 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %storemerge37, i64 %storemerge526.us
  %scevgep57 = getelementptr [512 x [512 x double]]* @B, i64 0, i64 %storemerge526.us, i64 %storemerge431.us
  %1 = load double* %scevgep58, align 8
  %2 = load double* %scevgep57, align 8
  %3 = fmul double %1, %2
  %4 = fadd double %.tmp.029.us, %3
  %5 = add nsw i64 %storemerge526.us, 1
  %exitcond10 = icmp eq i64 %5, 512
  br i1 %exitcond10, label %bb4.us, label %bb2.us

bb.nph27.us:                                      ; preds = %bb5.preheader, %bb4.us
  %storemerge431.us = phi i64 [ %0, %bb4.us ], [ 0, %bb5.preheader ]
  %scevgep61 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %storemerge37, i64 %storemerge431.us
  store double 0.000000e+00, double* %scevgep61, align 8
  br label %bb2.us

bb6:                                              ; preds = %bb4.us
  %6 = add nsw i64 %storemerge37, 1
  %exitcond16 = icmp ne i64 %6, 512
  br i1 %exitcond16, label %bb5.preheader, label %bb14.preheader.preheader

bb14.preheader.preheader:                         ; preds = %bb6
  br label %bb14.preheader

bb5.preheader:                                    ; preds = %bb6, %bb.nph50.bb.nph50.split_crit_edge
  %storemerge37 = phi i64 [ 0, %bb.nph50.bb.nph50.split_crit_edge ], [ %6, %bb6 ]
  br label %bb.nph27.us

bb13.us:                                          ; preds = %bb11.us
  %.lcssa = phi double [ %11, %bb11.us ]
  store double %.lcssa, double* %scevgep54
  %7 = add nsw i64 %storemerge27.us, 1
  %exitcond = icmp eq i64 %7, 512
  br i1 %exitcond, label %bb15, label %bb.nph.us

bb11.us:                                          ; preds = %bb.nph.us, %bb11.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %11, %bb11.us ]
  %storemerge36.us = phi i64 [ 0, %bb.nph.us ], [ %12, %bb11.us ]
  %scevgep51 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %storemerge112, i64 %storemerge36.us
  %scevgep = getelementptr [512 x [512 x double]]* @D, i64 0, i64 %storemerge36.us, i64 %storemerge27.us
  %8 = load double* %scevgep51, align 8
  %9 = load double* %scevgep, align 8
  %10 = fmul double %8, %9
  %11 = fadd double %.tmp.0.us, %10
  %12 = add nsw i64 %storemerge36.us, 1
  %exitcond1 = icmp eq i64 %12, 512
  br i1 %exitcond1, label %bb13.us, label %bb11.us

bb.nph.us:                                        ; preds = %bb14.preheader, %bb13.us
  %storemerge27.us = phi i64 [ %7, %bb13.us ], [ 0, %bb14.preheader ]
  %scevgep54 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %storemerge112, i64 %storemerge27.us
  store double 0.000000e+00, double* %scevgep54, align 8
  br label %bb11.us

bb15:                                             ; preds = %bb13.us
  %13 = add nsw i64 %storemerge112, 1
  %exitcond6 = icmp ne i64 %13, 512
  br i1 %exitcond6, label %bb14.preheader, label %return

bb14.preheader:                                   ; preds = %bb14.preheader.preheader, %bb15
  %storemerge112 = phi i64 [ %13, %bb15 ], [ 0, %bb14.preheader.preheader ]
  br label %bb.nph.us

return:                                           ; preds = %bb15
  ret void
}
; CHECK: Valid Region for Scop: bb5.preheader => return
