; RUN: opt %loadPolly  %defaultOpts -polly-cloog -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@alpha = common global double 0.000000e+00
@beta = common global double 0.000000e+00
@x = common global [4000 x double] zeroinitializer, align 32
@A = common global [4000 x [4000 x double]] zeroinitializer, align 32
@y = common global [4000 x double] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@tmp = common global [4000 x double] zeroinitializer, align 32
@B = common global [4000 x [4000 x double]] zeroinitializer, align 32

define void @scop_func(i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb.nph10.split.us, label %return

bb.nph10.split.us:                                ; preds = %entry
  %1 = load double* @alpha, align 8
  %2 = load double* @beta, align 8
  br label %bb.nph.us

bb3.us:                                           ; preds = %bb1.us
  %.lcssa1 = phi double [ %13, %bb1.us ]
  %.lcssa = phi double [ %10, %bb1.us ]
  store double %.lcssa, double* %scevgep17
  %3 = fmul double %.lcssa, %1
  %4 = fmul double %.lcssa1, %2
  %5 = fadd double %3, %4
  store double %5, double* %scevgep18, align 8
  %6 = add nsw i64 %storemerge6.us, 1
  %exitcond = icmp eq i64 %6, %n
  br i1 %exitcond, label %return.loopexit, label %bb.nph.us

bb1.us:                                           ; preds = %bb.nph.us, %bb1.us
  %.tmp3.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %13, %bb1.us ]
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %10, %bb1.us ]
  %storemerge12.us = phi i64 [ 0, %bb.nph.us ], [ %14, %bb1.us ]
  %scevgep13 = getelementptr [4000 x [4000 x double]]* @A, i64 0, i64 %storemerge6.us, i64 %storemerge12.us
  %scevgep = getelementptr [4000 x [4000 x double]]* @B, i64 0, i64 %storemerge6.us, i64 %storemerge12.us
  %scevgep12 = getelementptr [4000 x double]* @x, i64 0, i64 %storemerge12.us
  %7 = load double* %scevgep13, align 8
  %8 = load double* %scevgep12, align 8
  %9 = fmul double %7, %8
  %10 = fadd double %9, %.tmp.0.us
  %11 = load double* %scevgep, align 8
  %12 = fmul double %11, %8
  %13 = fadd double %12, %.tmp3.0.us
  %14 = add nsw i64 %storemerge12.us, 1
  %exitcond2 = icmp eq i64 %14, %n
  br i1 %exitcond2, label %bb3.us, label %bb1.us

bb.nph.us:                                        ; preds = %bb3.us, %bb.nph10.split.us
  %storemerge6.us = phi i64 [ 0, %bb.nph10.split.us ], [ %6, %bb3.us ]
  %scevgep18 = getelementptr [4000 x double]* @y, i64 0, i64 %storemerge6.us
  %scevgep17 = getelementptr [4000 x double]* @tmp, i64 0, i64 %storemerge6.us
  store double 0.000000e+00, double* %scevgep17, align 8
  store double 0.000000e+00, double* %scevgep18, align 8
  br label %bb1.us

return.loopexit:                                  ; preds = %bb3.us
  br label %return

return:                                           ; preds = %return.loopexit, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
