; RUN: opt %loadPolly  %defaultOpts -polly-analyze-ir  -print-top-scop-only -analyze %s | FileCheck %s
; XFAIL: *
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@float_n = global double 0x41B32863F6028F5C
@eps = global double 5.000000e-03
@data = common global [501 x [501 x double]] zeroinitializer, align 32
@symmat = common global [501 x [501 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@mean = common global [501 x double] zeroinitializer, align 32
@stddev = common global [501 x double] zeroinitializer, align 32

define void @scop_func() nounwind {
bb.nph33.bb.nph33.split_crit_edge:
  br label %bb2.preheader

bb1:                                              ; preds = %bb2.preheader, %bb1
  %indvar45 = phi i64 [ %tmp57, %bb1 ], [ 0, %bb2.preheader ]
  %tmp51 = add i64 %indvar45, 1
  %scevgep53 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp50, i64 %tmp51
  %tmp44 = add i64 %indvar45, 1
  %scevgep54 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp44
  %scevgep49 = getelementptr [501 x double]* @stddev, i64 0, i64 %tmp44
  %tmp57 = add i64 %indvar45, 1
  %0 = load double* %scevgep53, align 8
  %1 = load double* %scevgep54, align 8
  %2 = fsub double %0, %1
  store double %2, double* %scevgep53, align 8
  %3 = load double* @float_n, align 8
  %4 = tail call double @sqrt(double %3) nounwind readonly
  %5 = load double* %scevgep49, align 8
  %6 = fmul double %4, %5
  %7 = fdiv double %2, %6
  store double %7, double* %scevgep53, align 8
  %exitcond43 = icmp eq i64 %tmp57, 500
  br i1 %exitcond43, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %tmp56 = add i64 %indvar50, 1
  %exitcond49 = icmp eq i64 %tmp56, 500
  br i1 %exitcond49, label %bb6.preheader, label %bb2.preheader

bb6.preheader:                                    ; preds = %bb3
  br label %bb6

bb2.preheader:                                    ; preds = %bb3, %bb.nph33.bb.nph33.split_crit_edge
  %indvar50 = phi i64 [ 0, %bb.nph33.bb.nph33.split_crit_edge ], [ %tmp56, %bb3 ]
  %tmp50 = add i64 %indvar50, 1
  br label %bb1

bb6:                                              ; preds = %bb6.preheader, %bb12
  %indvar3 = phi i64 [ 0, %bb6.preheader ], [ %indvar.next, %bb12 ]
  %tmp25 = mul i64 %indvar3, 502
  %tmp26 = add i64 %tmp25, 2
  %tmp30 = add i64 %tmp25, 1
  %tmp33 = add i64 %indvar3, 2
  %tmp36 = mul i64 %indvar3, -1
  %tmp12 = add i64 %tmp36, 499
  %tmp38 = add i64 %indvar3, 1
  %scevgep42 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp30
  store double 1.000000e+00, double* %scevgep42, align 8
  br i1 false, label %bb12, label %bb.nph12.bb.nph12.split_crit_edge

bb.nph12.bb.nph12.split_crit_edge:                ; preds = %bb6
  br label %bb.nph

bb.nph:                                           ; preds = %bb10, %bb.nph12.bb.nph12.split_crit_edge
  %indvar6 = phi i64 [ %indvar.next7, %bb10 ], [ 0, %bb.nph12.bb.nph12.split_crit_edge ]
  %tmp27 = add i64 %tmp26, %indvar6
  %scevgep23 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp27
  %tmp29 = add i64 %indvar6, 2
  %scevgep20 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 %tmp29, i64 %tmp30
  %tmp34 = add i64 %tmp33, %indvar6
  store double 0.000000e+00, double* %scevgep23, align 8
  br label %bb8

bb8:                                              ; preds = %bb8, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp, %bb8 ]
  %8 = phi double [ 0.000000e+00, %bb.nph ], [ %12, %bb8 ]
  %tmp32 = add i64 %indvar, 1
  %scevgep = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp32, i64 %tmp34
  %scevgep41 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp32, i64 %tmp38
  %tmp = add i64 %indvar, 1
  %9 = load double* %scevgep41, align 8
  %10 = load double* %scevgep, align 8
  %11 = fmul double %9, %10
  %12 = fadd double %8, %11
  %exitcond1 = icmp eq i64 %tmp, 500
  br i1 %exitcond1, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %.lcssa = phi double [ %12, %bb8 ]
  store double %.lcssa, double* %scevgep23
  store double %.lcssa, double* %scevgep20, align 8
  %indvar.next7 = add i64 %indvar6, 1
  %exitcond = icmp eq i64 %indvar.next7, %tmp12
  br i1 %exitcond, label %bb12.loopexit, label %bb.nph

bb12.loopexit:                                    ; preds = %bb10
  br label %bb12

bb12:                                             ; preds = %bb12.loopexit, %bb6
  %indvar.next = add i64 %indvar3, 1
  %exitcond24 = icmp eq i64 %indvar.next, 499
  br i1 %exitcond24, label %return, label %bb6

return:                                           ; preds = %bb12
  ret void
}

declare double @sqrt(double) nounwind readonly
