; RUN: opt %loadPolly  %defaultOpts -polly-analyze-ir  -print-top-scop-only -analyze %s | FileCheck %s
; XFAIL: *
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

define void @scop_func(i64 %m, i64 %n) nounwind {
entry:
  %0 = icmp slt i64 %m, 1
  br i1 %0, label %bb10.preheader, label %bb.nph44

bb.nph44:                                         ; preds = %entry
  %1 = icmp slt i64 %n, 1
  %2 = load double* @float_n, align 8
  br i1 %1, label %bb3.us.preheader, label %bb.nph36.preheader

bb3.us.preheader:                                 ; preds = %bb.nph44
  br label %bb3.us

bb.nph36.preheader:                               ; preds = %bb.nph44
  br label %bb.nph36

bb3.us:                                           ; preds = %bb3.us.preheader, %bb3.us
  %indvar = phi i64 [ %tmp, %bb3.us ], [ 0, %bb3.us.preheader ]
  %tmp45 = add i64 %indvar, 2
  %tmp13 = add i64 %indvar, 1
  %scevgep = getelementptr [501 x double]* @mean, i64 0, i64 %tmp13
  %tmp = add i64 %indvar, 1
  %3 = fdiv double 0.000000e+00, %2
  store double %3, double* %scevgep, align 8
  %4 = icmp sgt i64 %tmp45, %m
  br i1 %4, label %bb10.preheader.loopexit1, label %bb3.us

bb.nph36:                                         ; preds = %bb.nph36.preheader, %bb3
  %indvar94 = phi i64 [ %tmp100, %bb3 ], [ 0, %bb.nph36.preheader ]
  %tmp8 = add i64 %indvar94, 1
  %tmp102 = add i64 %indvar94, 2
  %scevgep103 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp8
  %tmp100 = add i64 %indvar94, 1
  store double 0.000000e+00, double* %scevgep103, align 8
  br label %bb1

bb1:                                              ; preds = %bb1, %bb.nph36
  %indvar91 = phi i64 [ 0, %bb.nph36 ], [ %tmp99, %bb1 ]
  %5 = phi double [ 0.000000e+00, %bb.nph36 ], [ %7, %bb1 ]
  %tmp7 = add i64 %indvar91, 1
  %scevgep97 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp7, i64 %tmp8
  %tmp98 = add i64 %indvar91, 2
  %tmp99 = add i64 %indvar91, 1
  %6 = load double* %scevgep97, align 8
  %7 = fadd double %5, %6
  %8 = icmp sgt i64 %tmp98, %n
  br i1 %8, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %.lcssa = phi double [ %7, %bb1 ]
  %9 = fdiv double %.lcssa, %2
  store double %9, double* %scevgep103, align 8
  %10 = icmp sgt i64 %tmp102, %m
  br i1 %10, label %bb10.preheader.loopexit, label %bb.nph36

bb10.preheader.loopexit:                          ; preds = %bb3
  br label %bb10.preheader

bb10.preheader.loopexit1:                         ; preds = %bb3.us
  br label %bb10.preheader

bb10.preheader:                                   ; preds = %bb10.preheader.loopexit1, %bb10.preheader.loopexit, %entry
  %11 = icmp slt i64 %n, 1
  br i1 %11, label %bb19.preheader, label %bb.nph33

bb7:                                              ; preds = %bb8.preheader, %bb7
  %indvar77 = phi i64 [ %tmp87, %bb7 ], [ 0, %bb8.preheader ]
  %tmp21 = add i64 %indvar77, 1
  %scevgep83 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp20, i64 %tmp21
  %tmp85 = add i64 %indvar77, 2
  %tmp16 = add i64 %indvar77, 1
  %scevgep84 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp16
  %tmp87 = add i64 %indvar77, 1
  %12 = load double* %scevgep83, align 8
  %13 = load double* %scevgep84, align 8
  %14 = fsub double %12, %13
  store double %14, double* %scevgep83, align 8
  %15 = icmp sgt i64 %tmp85, %m
  br i1 %15, label %bb9, label %bb7

bb9:                                              ; preds = %bb7
  %16 = icmp sgt i64 %tmp89, %n
  br i1 %16, label %bb19.preheader.loopexit, label %bb8.preheader

bb.nph33:                                         ; preds = %bb10.preheader
  br i1 %0, label %return, label %bb8.preheader.preheader

bb8.preheader.preheader:                          ; preds = %bb.nph33
  br label %bb8.preheader

bb8.preheader:                                    ; preds = %bb8.preheader.preheader, %bb9
  %indvar79 = phi i64 [ %tmp86, %bb9 ], [ 0, %bb8.preheader.preheader ]
  %tmp20 = add i64 %indvar79, 1
  %tmp89 = add i64 %indvar79, 2
  %tmp86 = add i64 %indvar79, 1
  br label %bb7

bb19.preheader.loopexit:                          ; preds = %bb9
  br label %bb19.preheader

bb19.preheader:                                   ; preds = %bb19.preheader.loopexit, %bb10.preheader
  br i1 %0, label %return, label %bb17.preheader.preheader

bb17.preheader.preheader:                         ; preds = %bb19.preheader
  br label %bb17.preheader

bb.nph13:                                         ; preds = %bb17.preheader
  br i1 %11, label %bb16.us.preheader, label %bb.nph13.bb.nph13.split_crit_edge

bb16.us.preheader:                                ; preds = %bb.nph13
  br label %bb16.us

bb.nph13.bb.nph13.split_crit_edge:                ; preds = %bb.nph13
  br label %bb.nph

bb16.us:                                          ; preds = %bb16.us.preheader, %bb16.us
  %indvar48 = phi i64 [ %indvar.next49, %bb16.us ], [ 0, %bb16.us.preheader ]
  %tmp57 = add i64 %tmp56, %indvar48
  %scevgep57 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp57
  %tmp59 = add i64 %indvar48, 1
  %scevgep52 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 %tmp59, i64 %tmp56
  %tmp54 = add i64 %tmp61, %indvar48
  store double 0.000000e+00, double* %scevgep57, align 8
  store double 0.000000e+00, double* %scevgep52, align 8
  %17 = icmp sgt i64 %tmp54, %m
  %indvar.next49 = add i64 %indvar48, 1
  br i1 %17, label %bb18.loopexit2, label %bb16.us

bb.nph:                                           ; preds = %bb16, %bb.nph13.bb.nph13.split_crit_edge
  %indvar62 = phi i64 [ 0, %bb.nph13.bb.nph13.split_crit_edge ], [ %indvar.next63, %bb16 ]
  %tmp72 = add i64 %tmp61, %indvar62
  %tmp64 = add i64 %indvar62, 1
  %scevgep74 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 %tmp64, i64 %tmp56
  %tmp69 = add i64 %tmp56, %indvar62
  %scevgep76 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp69
  %tmp74 = add i64 %storemerge214, %indvar62
  store double 0.000000e+00, double* %scevgep76, align 8
  br label %bb14

bb14:                                             ; preds = %bb14, %bb.nph
  %indvar59 = phi i64 [ 0, %bb.nph ], [ %tmp68, %bb14 ]
  %18 = phi double [ 0.000000e+00, %bb.nph ], [ %22, %bb14 ]
  %tmp71 = add i64 %indvar59, 1
  %scevgep65 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp71, i64 %tmp74
  %scevgep66 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp71, i64 %storemerge214
  %tmp67 = add i64 %indvar59, 2
  %tmp68 = add i64 %indvar59, 1
  %19 = load double* %scevgep66, align 8
  %20 = load double* %scevgep65, align 8
  %21 = fmul double %19, %20
  %22 = fadd double %18, %21
  %23 = icmp sgt i64 %tmp67, %n
  br i1 %23, label %bb16, label %bb14

bb16:                                             ; preds = %bb14
  %.lcssa24 = phi double [ %22, %bb14 ]
  store double %.lcssa24, double* %scevgep76
  store double %.lcssa24, double* %scevgep74, align 8
  %24 = icmp sgt i64 %tmp72, %m
  %indvar.next63 = add i64 %indvar62, 1
  br i1 %24, label %bb18.loopexit, label %bb.nph

bb18.loopexit:                                    ; preds = %bb16
  br label %bb18

bb18.loopexit2:                                   ; preds = %bb16.us
  br label %bb18

bb18:                                             ; preds = %bb18.loopexit2, %bb18.loopexit, %bb17.preheader
  %indvar.next = add i64 %indvar27, 1
  %exitcond = icmp eq i64 %indvar.next, %m
  br i1 %exitcond, label %return.loopexit, label %bb17.preheader

bb17.preheader:                                   ; preds = %bb17.preheader.preheader, %bb18
  %indvar27 = phi i64 [ 0, %bb17.preheader.preheader ], [ %indvar.next, %bb18 ]
  %tmp55 = mul i64 %indvar27, 502
  %tmp56 = add i64 %tmp55, 1
  %tmp61 = add i64 %indvar27, 2
  %storemerge214 = add i64 %indvar27, 1
  br i1 false, label %bb18, label %bb.nph13

return.loopexit:                                  ; preds = %bb18
  br label %return

return:                                           ; preds = %return.loopexit, %bb19.preheader, %bb.nph33
  ret void
}
