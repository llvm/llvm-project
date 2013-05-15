; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [128 x [128 x [128 x double]]] zeroinitializer, align 32
@C4 = common global [128 x [128 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@sum = common global [128 x [128 x [128 x double]]] zeroinitializer, align 32

define void @scop_func() nounwind {
bb.nph50.bb.nph50.split_crit_edge:
  br label %bb11.preheader

bb5.us:                                           ; preds = %bb3.us
  %.lcssa = phi double [ %4, %bb3.us ]
  store double %.lcssa, double* %scevgep54
  %0 = add nsw i64 %storemerge26.us, 1
  %exitcond = icmp eq i64 %0, 128
  br i1 %exitcond, label %bb8.loopexit, label %bb.nph.us

bb3.us:                                           ; preds = %bb.nph.us, %bb3.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %4, %bb3.us ]
  %storemerge45.us = phi i64 [ 0, %bb.nph.us ], [ %5, %bb3.us ]
  %scevgep51 = getelementptr [128 x [128 x [128 x double]]]* @A, i64 0, i64 %storemerge30, i64 %storemerge113, i64 %storemerge45.us
  %scevgep = getelementptr [128 x [128 x double]]* @C4, i64 0, i64 %storemerge45.us, i64 %storemerge26.us
  %1 = load double* %scevgep51, align 8
  %2 = load double* %scevgep, align 8
  %3 = fmul double %1, %2
  %4 = fadd double %.tmp.0.us, %3
  %5 = add nsw i64 %storemerge45.us, 1
  %exitcond1 = icmp eq i64 %5, 128
  br i1 %exitcond1, label %bb5.us, label %bb3.us

bb.nph.us:                                        ; preds = %bb6.preheader, %bb5.us
  %storemerge26.us = phi i64 [ %0, %bb5.us ], [ 0, %bb6.preheader ]
  %scevgep54 = getelementptr [128 x [128 x [128 x double]]]* @sum, i64 0, i64 %storemerge30, i64 %storemerge113, i64 %storemerge26.us
  store double 0.000000e+00, double* %scevgep54, align 8
  br label %bb3.us

bb8.loopexit:                                     ; preds = %bb5.us
  br label %bb8

bb8:                                              ; preds = %bb8.loopexit, %bb8
  %storemerge311 = phi i64 [ %7, %bb8 ], [ 0, %bb8.loopexit ]
  %scevgep57 = getelementptr [128 x [128 x [128 x double]]]* @sum, i64 0, i64 %storemerge30, i64 %storemerge113, i64 %storemerge311
  %scevgep56 = getelementptr [128 x [128 x [128 x double]]]* @A, i64 0, i64 %storemerge30, i64 %storemerge113, i64 %storemerge311
  %6 = load double* %scevgep57, align 8
  store double %6, double* %scevgep56, align 8
  %7 = add nsw i64 %storemerge311, 1
  %exitcond6 = icmp eq i64 %7, 128
  br i1 %exitcond6, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %8 = add nsw i64 %storemerge113, 1
  %exitcond9 = icmp ne i64 %8, 128
  br i1 %exitcond9, label %bb6.preheader, label %bb12

bb6.preheader:                                    ; preds = %bb11.preheader, %bb10
  %storemerge113 = phi i64 [ %8, %bb10 ], [ 0, %bb11.preheader ]
  br label %bb.nph.us

bb12:                                             ; preds = %bb10
  %9 = add nsw i64 %storemerge30, 1
  %exitcond14 = icmp ne i64 %9, 128
  br i1 %exitcond14, label %bb11.preheader, label %return

bb11.preheader:                                   ; preds = %bb12, %bb.nph50.bb.nph50.split_crit_edge
  %storemerge30 = phi i64 [ 0, %bb.nph50.bb.nph50.split_crit_edge ], [ %9, %bb12 ]
  br label %bb6.preheader

return:                                           ; preds = %bb12
  ret void
}
; CHECK: Valid Region for Scop: bb11.preheader => return 
