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

define void @scop_func(i64 %m, i64 %n) nounwind {
entry:
  %0 = icmp slt i64 %n, 1
  %1 = icmp slt i64 %m, 1
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %bb13.preheader, label %bb2.preheader.preheader

bb2.preheader.preheader:                          ; preds = %entry
  br label %bb2.preheader

bb1:                                              ; preds = %bb2.preheader, %bb1
  %indvar52 = phi i64 [ %tmp63, %bb1 ], [ 0, %bb2.preheader ]
  %tmp9 = add i64 %indvar52, 1
  %scevgep59 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp8, i64 %tmp9
  %tmp61 = add i64 %indvar52, 2
  %tmp3 = add i64 %indvar52, 1
  %scevgep60 = getelementptr [501 x double]* @mean, i64 0, i64 %tmp3
  %scevgep55 = getelementptr [501 x double]* @stddev, i64 0, i64 %tmp3
  %tmp63 = add i64 %indvar52, 1
  %2 = load double* %scevgep59, align 8
  %3 = load double* %scevgep60, align 8
  %4 = fsub double %2, %3
  store double %4, double* %scevgep59, align 8
  %5 = load double* @float_n, align 8
  %6 = tail call double @sqrt(double %5) nounwind readonly
  %7 = load double* %scevgep55, align 8
  %8 = fmul double %6, %7
  %9 = fdiv double %4, %8
  store double %9, double* %scevgep59, align 8
  %10 = icmp sgt i64 %tmp61, %m
  br i1 %10, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %11 = icmp sgt i64 %tmp65, %n
  br i1 %11, label %bb13.preheader.loopexit, label %bb2.preheader

bb2.preheader:                                    ; preds = %bb2.preheader.preheader, %bb3
  %indvar56 = phi i64 [ %tmp62, %bb3 ], [ 0, %bb2.preheader.preheader ]
  %tmp8 = add i64 %indvar56, 1
  %tmp65 = add i64 %indvar56, 2
  %tmp62 = add i64 %indvar56, 1
  br label %bb1

bb13.preheader.loopexit:                          ; preds = %bb3
  br label %bb13.preheader

bb13.preheader:                                   ; preds = %bb13.preheader.loopexit, %entry
  %12 = add nsw i64 %m, -1
  %13 = icmp slt i64 %12, 1
  br i1 %13, label %return, label %bb6.preheader

bb6.preheader:                                    ; preds = %bb13.preheader
  %tmp = add i64 %m, -1
  br label %bb6

bb6:                                              ; preds = %bb6.preheader, %bb12
  %indvar14 = phi i64 [ 0, %bb6.preheader ], [ %indvar.next15, %bb12 ]
  %tmp35 = add i64 %indvar14, 3
  %tmp36 = trunc i64 %tmp35 to i32
  %tmp38 = add i64 %indvar14, 2
  %tmp39 = trunc i64 %tmp38 to i32
  %tmp46 = add i64 %indvar14, 1
  %scevgep49 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 0, i64 %tmp46
  %scevgep53 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 %tmp46, i64 0
  %tmp59 = mul i64 %indvar14, 502
  %tmp60 = add i64 %tmp59, 1
  %scevgep61 = getelementptr [501 x [501 x double]]* @symmat, i64 0, i64 1, i64 %tmp60
  store double 1.000000e+00, double* %scevgep61, align 8
  %14 = icmp sgt i64 %tmp38, %m
  br i1 %14, label %bb12, label %bb.nph12

bb.nph12:                                         ; preds = %bb6
  br i1 %0, label %bb10.us.preheader, label %bb.nph.preheader

bb10.us.preheader:                                ; preds = %bb.nph12
  br label %bb10.us

bb.nph.preheader:                                 ; preds = %bb.nph12
  br label %bb.nph

bb10.us:                                          ; preds = %bb10.us.preheader, %bb10.us
  %indvar = phi i32 [ %indvar.next, %bb10.us ], [ 0, %bb10.us.preheader ]
  %storemerge2.us = add i32 %tmp36, %indvar
  %storemerge28.us = add i32 %tmp39, %indvar
  %tmp55 = sext i32 %storemerge28.us to i64
  %tmp56 = mul i64 %tmp55, 501
  %scevgep57 = getelementptr double* %scevgep49, i64 %tmp56
  %scevgep58 = getelementptr double* %scevgep53, i64 %tmp55
  store double 0.000000e+00, double* %scevgep58, align 8
  store double 0.000000e+00, double* %scevgep57, align 8
  %15 = sext i32 %storemerge2.us to i64
  %16 = icmp sgt i64 %15, %m
  %indvar.next = add i32 %indvar, 1
  br i1 %16, label %bb12.loopexit1, label %bb10.us

bb.nph:                                           ; preds = %bb.nph.preheader, %bb10
  %indvar41 = phi i32 [ %indvar.next42, %bb10 ], [ 0, %bb.nph.preheader ]
  %storemerge2 = add i32 %tmp36, %indvar41
  %storemerge28 = add i32 %tmp39, %indvar41
  %tmp50 = sext i32 %storemerge28 to i64
  %tmp51 = mul i64 %tmp50, 501
  %scevgep52 = getelementptr double* %scevgep49, i64 %tmp51
  %scevgep54 = getelementptr double* %scevgep53, i64 %tmp50
  %tmp21 = sext i32 %storemerge28 to i64
  store double 0.000000e+00, double* %scevgep54, align 8
  br label %bb8

bb8:                                              ; preds = %bb8, %bb.nph
  %indvar38 = phi i64 [ 0, %bb.nph ], [ %tmp40, %bb8 ]
  %17 = phi double [ 0.000000e+00, %bb.nph ], [ %21, %bb8 ]
  %tmp44 = add i64 %indvar38, 1
  %scevgep47 = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp44, i64 %tmp46
  %tmp48 = add i64 %indvar38, 2
  %tmp13 = add i64 %indvar38, 1
  %scevgep = getelementptr [501 x [501 x double]]* @data, i64 0, i64 %tmp13, i64 %tmp21
  %tmp40 = add i64 %indvar38, 1
  %18 = load double* %scevgep47, align 8
  %19 = load double* %scevgep, align 8
  %20 = fmul double %18, %19
  %21 = fadd double %17, %20
  %22 = icmp sgt i64 %tmp48, %n
  br i1 %22, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %.lcssa = phi double [ %21, %bb8 ]
  store double %.lcssa, double* %scevgep54
  store double %.lcssa, double* %scevgep52, align 8
  %23 = sext i32 %storemerge2 to i64
  %24 = icmp sgt i64 %23, %m
  %indvar.next42 = add i32 %indvar41, 1
  br i1 %24, label %bb12.loopexit, label %bb.nph

bb12.loopexit:                                    ; preds = %bb10
  br label %bb12

bb12.loopexit1:                                   ; preds = %bb10.us
  br label %bb12

bb12:                                             ; preds = %bb12.loopexit1, %bb12.loopexit, %bb6
  %indvar.next15 = add i64 %indvar14, 1
  %exitcond = icmp eq i64 %indvar.next15, %tmp
  br i1 %exitcond, label %return.loopexit, label %bb6

return.loopexit:                                  ; preds = %bb12
  br label %return

return:                                           ; preds = %return.loopexit, %bb13.preheader
  ret void
}

declare double @sqrt(double) nounwind readonly
