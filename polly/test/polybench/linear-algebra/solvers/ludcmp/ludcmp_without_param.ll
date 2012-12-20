; RUN: opt %loadPolly  %defaultOpts -polly-detect -polly-ast -analyze  %s | FileCheck %s
; region-simplify make polly fail to detect the canonical induction variable.
; XFAIL:*

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@x = common global [1025 x double] zeroinitializer, align 32
@b = common global [1025 x double] zeroinitializer, align 32
@a = common global [1025 x [1025 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@w = common global double 0.000000e+00
@y = common global [1025 x double] zeroinitializer, align 32

define void @scop_func() nounwind {
bb.nph76:
  store double 1.000000e+00, double* getelementptr inbounds ([1025 x double]* @b, i64 0, i64 0), align 32
  %w.promoted = load double* @w
  br label %bb5.preheader

bb.nph38:                                         ; preds = %bb5.preheader
  %0 = icmp sgt i64 %storemerge55, 0
  br i1 %0, label %bb.nph38.split.us, label %bb4.preheader

bb4.preheader:                                    ; preds = %bb.nph38
  br label %bb4

bb.nph38.split.us:                                ; preds = %bb.nph38
  br label %bb.nph30.us

bb4.us:                                           ; preds = %bb2.us
  %.lcssa62 = phi double [ %7, %bb2.us ]
  %1 = load double* %scevgep109, align 8
  %2 = fdiv double %.lcssa62, %1
  store double %2, double* %scevgep141, align 8
  %exitcond70 = icmp eq i64 %tmp139, %tmp46
  br i1 %exitcond70, label %bb11.loopexit.loopexit1, label %bb.nph30.us

bb2.us:                                           ; preds = %bb.nph30.us, %bb2.us
  %3 = phi double [ %9, %bb.nph30.us ], [ %7, %bb2.us ]
  %storemerge829.us = phi i64 [ 0, %bb.nph30.us ], [ %8, %bb2.us ]
  %scevgep134 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp95, i64 %storemerge829.us
  %scevgep129 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge829.us, i64 %storemerge55
  %4 = load double* %scevgep134, align 8
  %5 = load double* %scevgep129, align 8
  %6 = fmul double %4, %5
  %7 = fsub double %3, %6
  %8 = add nsw i64 %storemerge829.us, 1
  %exitcond63 = icmp eq i64 %8, %storemerge55
  br i1 %exitcond63, label %bb4.us, label %bb2.us

bb.nph30.us:                                      ; preds = %bb4.us, %bb.nph38.split.us
  %indvar130 = phi i64 [ %tmp139, %bb4.us ], [ 0, %bb.nph38.split.us ]
  %tmp92 = add i64 %indvar130, 1
  %scevgep141 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp92, i64 %tmp86
  %tmp95 = add i64 %storemerge533, %indvar130
  %tmp139 = add i64 %indvar130, 1
  %9 = load double* %scevgep141, align 8
  br label %bb2.us

bb4:                                              ; preds = %bb4.preheader, %bb4
  %indvar145 = phi i64 [ %indvar.next146, %bb4 ], [ 0, %bb4.preheader ]
  %tmp99 = add i64 %indvar145, 1
  %scevgep150 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp99, i64 %tmp86
  %10 = load double* %scevgep150, align 8
  %11 = load double* %scevgep109, align 8
  %12 = fdiv double %10, %11
  store double %12, double* %scevgep150, align 8
  %indvar.next146 = add i64 %indvar145, 1
  %exitcond58 = icmp eq i64 %indvar.next146, %tmp46
  br i1 %exitcond58, label %bb11.loopexit.loopexit, label %bb4

bb.nph51:                                         ; preds = %bb11.loopexit
  br i1 false, label %bb10.us.preheader, label %bb.nph42.preheader

bb10.us.preheader:                                ; preds = %bb.nph51
  br label %bb10.us

bb.nph42.preheader:                               ; preds = %bb.nph51
  br label %bb.nph42

bb10.us:                                          ; preds = %bb10.us.preheader, %bb10.us
  %indvar114 = phi i64 [ %indvar.next115, %bb10.us ], [ 0, %bb10.us.preheader ]
  %tmp88 = add i64 %tmp87, %indvar114
  %scevgep121 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1, i64 %tmp88
  %13 = load double* %scevgep121, align 8
  store double %13, double* %scevgep121, align 8
  %indvar.next115 = add i64 %indvar114, 1
  %exitcond80 = icmp eq i64 %indvar.next115, %tmp46
  br i1 %exitcond80, label %bb13.loopexit.loopexit2, label %bb10.us

bb.nph42:                                         ; preds = %bb.nph42.preheader, %bb10
  %indvar155 = phi i64 [ %indvar.next156, %bb10 ], [ 0, %bb.nph42.preheader ]
  %tmp102 = add i64 %tmp87, %indvar155
  %scevgep173 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1, i64 %tmp102
  %tmp104 = add i64 %storemerge533, %indvar155
  %14 = load double* %scevgep173, align 8
  br label %bb8

bb8:                                              ; preds = %bb8, %bb.nph42
  %w.tmp.043 = phi double [ %14, %bb.nph42 ], [ %18, %bb8 ]
  %storemerge741 = phi i64 [ 0, %bb.nph42 ], [ %19, %bb8 ]
  %scevgep159 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge741, i64 %tmp104
  %scevgep160 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge533, i64 %storemerge741
  %15 = load double* %scevgep160, align 8
  %16 = load double* %scevgep159, align 8
  %17 = fmul double %15, %16
  %18 = fsub double %w.tmp.043, %17
  %19 = add nsw i64 %storemerge741, 1
  %exitcond41 = icmp eq i64 %19, %storemerge533
  br i1 %exitcond41, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %.lcssa37 = phi double [ %18, %bb8 ]
  store double %.lcssa37, double* %scevgep173, align 8
  %indvar.next156 = add i64 %indvar155, 1
  %exitcond47 = icmp eq i64 %indvar.next156, %tmp46
  br i1 %exitcond47, label %bb13.loopexit.loopexit, label %bb.nph42

bb11.loopexit.loopexit:                           ; preds = %bb4
  %.lcssa55 = phi double [ %10, %bb4 ]
  br label %bb11.loopexit

bb11.loopexit.loopexit1:                          ; preds = %bb4.us
  %.lcssa62.lcssa = phi double [ %.lcssa62, %bb4.us ]
  br label %bb11.loopexit

bb11.loopexit:                                    ; preds = %bb11.loopexit.loopexit1, %bb11.loopexit.loopexit, %bb5.preheader
  %w.tmp.077 = phi double [ %w.tmp.1, %bb5.preheader ], [ %.lcssa55, %bb11.loopexit.loopexit ], [ %.lcssa62.lcssa, %bb11.loopexit.loopexit1 ]
  br i1 false, label %bb13.loopexit, label %bb.nph51

bb13.loopexit.loopexit:                           ; preds = %bb10
  %.lcssa37.lcssa = phi double [ %.lcssa37, %bb10 ]
  br label %bb13.loopexit

bb13.loopexit.loopexit2:                          ; preds = %bb10.us
  %.lcssa77 = phi double [ %13, %bb10.us ]
  br label %bb13.loopexit

bb13.loopexit:                                    ; preds = %bb13.loopexit.loopexit2, %bb13.loopexit.loopexit, %bb11.loopexit
  %w.tmp.2 = phi double [ %w.tmp.077, %bb11.loopexit ], [ %.lcssa37.lcssa, %bb13.loopexit.loopexit ], [ %.lcssa77, %bb13.loopexit.loopexit2 ]
  %indvar.next39 = add i64 %storemerge55, 1
  %exitcond85 = icmp ne i64 %indvar.next39, 1024
  br i1 %exitcond85, label %bb5.preheader, label %bb.nph25

bb5.preheader:                                    ; preds = %bb13.loopexit, %bb.nph76
  %storemerge55 = phi i64 [ %indvar.next39, %bb13.loopexit ], [ 0, %bb.nph76 ]
  %w.tmp.1 = phi double [ %w.promoted, %bb.nph76 ], [ %w.tmp.2, %bb13.loopexit ]
  %tmp86 = mul i64 %storemerge55, 1026
  %tmp87 = add i64 %tmp86, 1
  %tmp90 = mul i64 %storemerge55, -1
  %tmp46 = add i64 %tmp90, 1024
  %storemerge533 = add i64 %storemerge55, 1
  %scevgep109 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 0, i64 %tmp86
  br i1 false, label %bb11.loopexit, label %bb.nph38

bb.nph25:                                         ; preds = %bb13.loopexit
  %w.tmp.2.lcssa = phi double [ %w.tmp.2, %bb13.loopexit ]
  store double %w.tmp.2.lcssa, double* @w
  store double 1.000000e+00, double* getelementptr inbounds ([1025 x double]* @y, i64 0, i64 0), align 32
  br label %bb.nph19

bb.nph19:                                         ; preds = %bb18, %bb.nph25
  %indvar102 = phi i64 [ 0, %bb.nph25 ], [ %tmp, %bb18 ]
  %tmp29 = add i64 %indvar102, 1
  %scevgep111 = getelementptr [1025 x double]* @b, i64 0, i64 %tmp29
  %scevgep110 = getelementptr [1025 x double]* @y, i64 0, i64 %tmp29
  %tmp = add i64 %indvar102, 1
  %20 = load double* %scevgep111, align 8
  br label %bb16

bb16:                                             ; preds = %bb16, %bb.nph19
  %21 = phi double [ %20, %bb.nph19 ], [ %25, %bb16 ]
  %storemerge418 = phi i64 [ 0, %bb.nph19 ], [ %26, %bb16 ]
  %scevgep106 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp29, i64 %storemerge418
  %scevgep105 = getelementptr [1025 x double]* @y, i64 0, i64 %storemerge418
  %22 = load double* %scevgep106, align 8
  %23 = load double* %scevgep105, align 8
  %24 = fmul double %22, %23
  %25 = fsub double %21, %24
  %26 = add nsw i64 %storemerge418, 1
  %exitcond = icmp eq i64 %26, %tmp29
  br i1 %exitcond, label %bb18, label %bb16

bb18:                                             ; preds = %bb16
  %.lcssa28 = phi double [ %25, %bb16 ]
  store double %.lcssa28, double* %scevgep110, align 8
  %exitcond32 = icmp eq i64 %tmp, 1024
  br i1 %exitcond32, label %bb.nph14, label %bb.nph19

bb.nph14:                                         ; preds = %bb18
  %.lcssa28.lcssa = phi double [ %.lcssa28, %bb18 ]
  store double %.lcssa28.lcssa, double* @w
  %27 = load double* getelementptr inbounds ([1025 x double]* @y, i64 0, i64 1024), align 32
  %28 = load double* getelementptr inbounds ([1025 x [1025 x double]]* @a, i64 0, i64 1024, i64 1024), align 32
  %29 = fdiv double %27, %28
  store double %29, double* getelementptr inbounds ([1025 x double]* @x, i64 0, i64 1024), align 32
  br label %bb.nph

bb.nph:                                           ; preds = %bb24, %bb.nph14
  %storemerge210 = phi i64 [ 0, %bb.nph14 ], [ %37, %bb24 ]
  %tmp14 = mul i64 %storemerge210, -1026
  %tmp15 = add i64 %tmp14, 1024
  %tmp18 = mul i64 %storemerge210, -1
  %tmp19 = add i64 %tmp18, 1024
  %tmp3 = add i64 %storemerge210, 1
  %tmp23 = add i64 %tmp18, 1023
  %scevgep100 = getelementptr [1025 x double]* @y, i64 0, i64 %tmp23
  %scevgep99 = getelementptr [1025 x double]* @x, i64 0, i64 %tmp23
  %tmp26 = add i64 %tmp14, 1023
  %scevgep97 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1023, i64 %tmp26
  %30 = load double* %scevgep100, align 8
  br label %bb22

bb22:                                             ; preds = %bb22, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb22 ]
  %w.tmp.0 = phi double [ %30, %bb.nph ], [ %34, %bb22 ]
  %tmp16 = add i64 %tmp15, %indvar
  %scevgep83 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1023, i64 %tmp16
  %tmp20 = add i64 %tmp19, %indvar
  %scevgep = getelementptr [1025 x double]* @x, i64 0, i64 %tmp20
  %31 = load double* %scevgep83, align 8
  %32 = load double* %scevgep, align 8
  %33 = fmul double %31, %32
  %34 = fsub double %w.tmp.0, %33
  %indvar.next = add i64 %indvar, 1
  %exitcond4 = icmp eq i64 %indvar.next, %tmp3
  br i1 %exitcond4, label %bb24, label %bb22

bb24:                                             ; preds = %bb22
  %.lcssa = phi double [ %34, %bb22 ]
  %35 = load double* %scevgep97, align 8
  %36 = fdiv double %.lcssa, %35
  store double %36, double* %scevgep99, align 8
  %37 = add nsw i64 %storemerge210, 1
  %exitcond13 = icmp eq i64 %37, 1024
  br i1 %exitcond13, label %return, label %bb.nph

return:                                           ; preds = %bb24
  %.lcssa.lcssa = phi double [ %.lcssa, %bb24 ]
  store double %.lcssa.lcssa, double* @w
  ret void
}
; CHECK: Valid Region for Scop: bb5.preheader => return 
