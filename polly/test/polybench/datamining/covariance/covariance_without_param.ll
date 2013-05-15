; RUN: opt %loadPolly  %defaultOpts -polly-detect -polly-ast -polly-codegen-scev -analyze  %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@float_n = global double 0x41B32863F6028F5C
@data = common global [501 x [501 x double]] zeroinitializer, align 32
@symmat = common global [501 x [501 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@mean = common global [501 x double] zeroinitializer, align 32

define void @scop_func() nounwind {
bb.nph44.bb.nph44.split_crit_edge:
  %0 = load double* @float_n, align 8
  br label %bb.nph36

bb.nph36:                                         ; preds = %bb3, %bb.nph44.bb.nph44.split_crit_edge
  %indvar77 = phi i64 [ 0, %bb.nph44.bb.nph44.split_crit_edge ], [ %tmp83, %bb3 ]
  %tmp48 = add i64 %indvar77, 1
  %scevgep85 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp48
  %tmp83 = add i64 %indvar77, 1
  store double 0.000000e+00, double* %scevgep85, align 8
  br label %bb1

bb1:                                              ; preds = %bb1, %bb.nph36
  %indvar73 = phi i64 [ 0, %bb.nph36 ], [ %tmp82, %bb1 ]
  %1 = phi double [ 0.000000e+00, %bb.nph36 ], [ %3, %bb1 ]
  %tmp47 = add i64 %indvar73, 1
  %scevgep80 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp47, i64 %tmp48
  %tmp82 = add i64 %indvar73, 1
  %2 = load double* %scevgep80, align 8
  %3 = fadd double %1, %2
  %exitcond42 = icmp eq i64 %tmp82, 500
  br i1 %exitcond42, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %.lcssa41 = phi double [ %3, %bb1 ]
  %4 = fdiv double %.lcssa41, %0
  store double %4, double* %scevgep85, align 8
  %exitcond46 = icmp eq i64 %tmp83, 500
  br i1 %exitcond46, label %bb8.preheader.preheader, label %bb.nph36

bb8.preheader.preheader:                          ; preds = %bb3
  br label %bb8.preheader

bb7:                                              ; preds = %bb8.preheader, %bb7
  %indvar59 = phi i64 [ %tmp70, %bb7 ], [ 0, %bb8.preheader ]
  %tmp39 = add i64 %indvar59, 1
  %scevgep66 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp38, i64 %tmp39
  %tmp = add i64 %indvar59, 1
  %scevgep67 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp
  %tmp70 = add i64 %indvar59, 1
  %5 = load double* %scevgep66, align 8
  %6 = load double* %scevgep67, align 8
  %7 = fsub double %5, %6
  store double %7, double* %scevgep66, align 8
  %exitcond33 = icmp eq i64 %tmp70, 500
  br i1 %exitcond33, label %bb9, label %bb7

bb9:                                              ; preds = %bb7
  %tmp69 = add i64 %indvar62, 1
  %exitcond37 = icmp eq i64 %tmp69, 500
  br i1 %exitcond37, label %bb17.preheader.preheader, label %bb8.preheader

bb17.preheader.preheader:                         ; preds = %bb9
  br label %bb17.preheader

bb8.preheader:                                    ; preds = %bb8.preheader.preheader, %bb9
  %indvar62 = phi i64 [ %tmp69, %bb9 ], [ 0, %bb8.preheader.preheader ]
  %tmp38 = add i64 %indvar62, 1
  br label %bb7

bb.nph13.bb.nph13.split_crit_edge:                ; preds = %bb17.preheader
  br label %bb.nph

bb.nph:                                           ; preds = %bb16, %bb.nph13.bb.nph13.split_crit_edge
  %indvar46 = phi i64 [ 0, %bb.nph13.bb.nph13.split_crit_edge ], [ %indvar.next47, %bb16 ]
  %tmp20 = add i64 %indvar46, 1
  %scevgep56 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 %tmp20, i64 %tmp22
  %tmp24 = add i64 %tmp22, %indvar46
  %scevgep58 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp24
  %tmp28 = add i64 %storemerge214, %indvar46
  store double 0.000000e+00, double* %scevgep58, align 8
  br label %bb14

bb14:                                             ; preds = %bb14, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp50, %bb14 ]
  %8 = phi double [ 0.000000e+00, %bb.nph ], [ %12, %bb14 ]
  %tmp26 = add i64 %indvar, 1
  %scevgep = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp26, i64 %tmp28
  %scevgep49 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp26, i64 %storemerge214
  %tmp50 = add i64 %indvar, 1
  %9 = load double* %scevgep49, align 8
  %10 = load double* %scevgep, align 8
  %11 = fmul double %9, %10
  %12 = fadd double %8, %11
  %exitcond1 = icmp eq i64 %tmp50, 500
  br i1 %exitcond1, label %bb16, label %bb14

bb16:                                             ; preds = %bb14
  %.lcssa = phi double [ %12, %bb14 ]
  store double %.lcssa, double* %scevgep58
  store double %.lcssa, double* %scevgep56, align 8
  %indvar.next47 = add i64 %indvar46, 1
  %exitcond = icmp eq i64 %indvar.next47, %tmp8
  br i1 %exitcond, label %bb18.loopexit, label %bb.nph

bb18.loopexit:                                    ; preds = %bb16
  br label %bb18

bb18:                                             ; preds = %bb18.loopexit, %bb17.preheader
  %indvar.next = add i64 %indvar2, 1
  %exitcond19 = icmp eq i64 %indvar.next, 500
  br i1 %exitcond19, label %return, label %bb17.preheader

bb17.preheader:                                   ; preds = %bb17.preheader.preheader, %bb18
  %indvar2 = phi i64 [ 0, %bb17.preheader.preheader ], [ %indvar.next, %bb18 ]
  %tmp21 = mul i64 %indvar2, 502
  %tmp22 = add i64 %tmp21, 1
  %storemerge214 = add i64 %indvar2, 1
  %tmp30 = mul i64 %indvar2, -1
  %tmp8 = add i64 %tmp30, 500
  br i1 false, label %bb18, label %bb.nph13.bb.nph13.split_crit_edge

return:                                           ; preds = %bb18
  ret void
}
; CHECK: for region: 'bb.nph36 => return' in function 'scop_func':
