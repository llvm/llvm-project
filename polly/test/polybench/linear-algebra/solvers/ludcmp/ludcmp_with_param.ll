; RUN: opt %loadPolly  %defaultOpts -polly-analyze-ir  -print-top-scop-only -analyze %s | FileCheck %s
; XFAIL: *
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

define void @scop_func(i64 %n) nounwind {
entry:
  store double 1.000000e+00, double* getelementptr inbounds ([1025 x double]* @b, i64 0, i64 0), align 32
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb.nph81, label %bb14

bb.nph43:                                         ; preds = %bb5.preheader
  %1 = icmp sgt i64 %storemerge60, 0
  br i1 %1, label %bb.nph43.split.us, label %bb4.preheader

bb4.preheader:                                    ; preds = %bb.nph43
  br label %bb4

bb.nph43.split.us:                                ; preds = %bb.nph43
  br label %bb.nph35.us

bb4.us:                                           ; preds = %bb2.us
  %.lcssa63 = phi double [ %9, %bb2.us ]
  %2 = load double* %scevgep110, align 8
  %3 = fdiv double %.lcssa63, %2
  store double %3, double* %scevgep148, align 8
  %4 = icmp sgt i64 %storemerge5.us, %n
  br i1 %4, label %bb11.loopexit.loopexit1, label %bb.nph35.us

bb2.us:                                           ; preds = %bb.nph35.us, %bb2.us
  %5 = phi double [ %11, %bb.nph35.us ], [ %9, %bb2.us ]
  %storemerge834.us = phi i64 [ 0, %bb.nph35.us ], [ %10, %bb2.us ]
  %scevgep141 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp96, i64 %storemerge834.us
  %scevgep136 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge834.us, i64 %storemerge60
  %6 = load double* %scevgep141, align 8
  %7 = load double* %scevgep136, align 8
  %8 = fmul double %6, %7
  %9 = fsub double %5, %8
  %10 = add nsw i64 %storemerge834.us, 1
  %exitcond64 = icmp eq i64 %10, %storemerge60
  br i1 %exitcond64, label %bb4.us, label %bb2.us

bb.nph35.us:                                      ; preds = %bb4.us, %bb.nph43.split.us
  %indvar137 = phi i64 [ %tmp146, %bb4.us ], [ 0, %bb.nph43.split.us ]
  %storemerge5.us = add i64 %tmp, %indvar137
  %tmp93 = add i64 %indvar137, 1
  %scevgep148 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp93, i64 %tmp87
  %tmp96 = add i64 %storemerge538, %indvar137
  %tmp146 = add i64 %indvar137, 1
  %11 = load double* %scevgep148, align 8
  br label %bb2.us

bb4:                                              ; preds = %bb4.preheader, %bb4
  %indvar152 = phi i64 [ %indvar.next153, %bb4 ], [ 0, %bb4.preheader ]
  %tmp99 = add i64 %indvar152, 1
  %scevgep157 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %tmp99, i64 %tmp87
  %storemerge5 = add i64 %tmp, %indvar152
  %12 = load double* %scevgep157, align 8
  %13 = load double* %scevgep110, align 8
  %14 = fdiv double %12, %13
  store double %14, double* %scevgep157, align 8
  %15 = icmp sgt i64 %storemerge5, %n
  %indvar.next153 = add i64 %indvar152, 1
  br i1 %15, label %bb11.loopexit.loopexit, label %bb4

bb.nph56:                                         ; preds = %bb11.loopexit
  br i1 false, label %bb10.us.preheader, label %bb.nph47.preheader

bb10.us.preheader:                                ; preds = %bb.nph56
  br label %bb10.us

bb.nph47.preheader:                               ; preds = %bb.nph56
  br label %bb.nph47

bb10.us:                                          ; preds = %bb10.us.preheader, %bb10.us
  %indvar122 = phi i64 [ %indvar.next123, %bb10.us ], [ 0, %bb10.us.preheader ]
  %storemerge6.us = add i64 %tmp, %indvar122
  %tmp89 = add i64 %tmp88, %indvar122
  %scevgep128 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1, i64 %tmp89
  %16 = load double* %scevgep128, align 8
  store double %16, double* %scevgep128, align 8
  %17 = icmp sgt i64 %storemerge6.us, %n
  %indvar.next123 = add i64 %indvar122, 1
  br i1 %17, label %bb13.loopexit.loopexit2, label %bb10.us

bb.nph47:                                         ; preds = %bb.nph47.preheader, %bb10
  %indvar162 = phi i64 [ %indvar.next163, %bb10 ], [ 0, %bb.nph47.preheader ]
  %storemerge6 = add i64 %tmp, %indvar162
  %tmp104 = add i64 %tmp88, %indvar162
  %scevgep180 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 1, i64 %tmp104
  %tmp107 = add i64 %storemerge538, %indvar162
  %18 = load double* %scevgep180, align 8
  br label %bb8

bb8:                                              ; preds = %bb8, %bb.nph47
  %w.tmp.048 = phi double [ %18, %bb.nph47 ], [ %22, %bb8 ]
  %storemerge746 = phi i64 [ 0, %bb.nph47 ], [ %23, %bb8 ]
  %scevgep166 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge746, i64 %tmp107
  %scevgep167 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge538, i64 %storemerge746
  %19 = load double* %scevgep167, align 8
  %20 = load double* %scevgep166, align 8
  %21 = fmul double %19, %20
  %22 = fsub double %w.tmp.048, %21
  %23 = add nsw i64 %storemerge746, 1
  %exitcond = icmp eq i64 %23, %smax
  br i1 %exitcond, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %.lcssa40 = phi double [ %22, %bb8 ]
  store double %.lcssa40, double* %scevgep180, align 8
  %24 = icmp sgt i64 %storemerge6, %n
  %indvar.next163 = add i64 %indvar162, 1
  br i1 %24, label %bb13.loopexit.loopexit, label %bb.nph47

bb11.loopexit.loopexit:                           ; preds = %bb4
  %.lcssa57 = phi double [ %12, %bb4 ]
  br label %bb11.loopexit

bb11.loopexit.loopexit1:                          ; preds = %bb4.us
  %.lcssa63.lcssa = phi double [ %.lcssa63, %bb4.us ]
  br label %bb11.loopexit

bb11.loopexit:                                    ; preds = %bb11.loopexit.loopexit1, %bb11.loopexit.loopexit, %bb5.preheader
  %w.tmp.082 = phi double [ %w.tmp.1, %bb5.preheader ], [ %.lcssa57, %bb11.loopexit.loopexit ], [ %.lcssa63.lcssa, %bb11.loopexit.loopexit1 ]
  %25 = icmp sgt i64 %storemerge538, %n
  br i1 %25, label %bb13.loopexit, label %bb.nph56

bb13.loopexit.loopexit:                           ; preds = %bb10
  %.lcssa40.lcssa = phi double [ %.lcssa40, %bb10 ]
  br label %bb13.loopexit

bb13.loopexit.loopexit2:                          ; preds = %bb10.us
  %.lcssa77 = phi double [ %16, %bb10.us ]
  br label %bb13.loopexit

bb13.loopexit:                                    ; preds = %bb13.loopexit.loopexit2, %bb13.loopexit.loopexit, %bb11.loopexit
  %w.tmp.2 = phi double [ %w.tmp.082, %bb11.loopexit ], [ %.lcssa40.lcssa, %bb13.loopexit.loopexit ], [ %.lcssa77, %bb13.loopexit.loopexit2 ]
  %indvar.next42 = add i64 %storemerge60, 1
  %exitcond84 = icmp ne i64 %indvar.next42, %n
  br i1 %exitcond84, label %bb5.preheader, label %bb13.bb14_crit_edge

bb13.bb14_crit_edge:                              ; preds = %bb13.loopexit
  %w.tmp.2.lcssa = phi double [ %w.tmp.2, %bb13.loopexit ]
  store double %w.tmp.2.lcssa, double* @w
  br label %bb14

bb.nph81:                                         ; preds = %entry
  %w.promoted = load double* @w
  br label %bb5.preheader

bb5.preheader:                                    ; preds = %bb.nph81, %bb13.loopexit
  %storemerge60 = phi i64 [ 0, %bb.nph81 ], [ %indvar.next42, %bb13.loopexit ]
  %w.tmp.1 = phi double [ %w.promoted, %bb.nph81 ], [ %w.tmp.2, %bb13.loopexit ]
  %tmp = add i64 %storemerge60, 2
  %tmp87 = mul i64 %storemerge60, 1026
  %tmp88 = add i64 %tmp87, 1
  %storemerge538 = add i64 %storemerge60, 1
  %scevgep110 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 0, i64 %tmp87
  %tmp44 = icmp sgt i64 %storemerge538, 1
  %smax = select i1 %tmp44, i64 %storemerge538, i64 1
  %26 = icmp sgt i64 %storemerge538, %n
  br i1 %26, label %bb11.loopexit, label %bb.nph43

bb14:                                             ; preds = %bb13.bb14_crit_edge, %entry
  store double 1.000000e+00, double* getelementptr inbounds ([1025 x double]* @y, i64 0, i64 0), align 32
  %27 = icmp slt i64 %n, 1
  br i1 %27, label %bb20, label %bb15.preheader

bb15.preheader:                                   ; preds = %bb14
  br label %bb15

bb15:                                             ; preds = %bb15.preheader, %bb18
  %indvar111 = phi i64 [ %28, %bb18 ], [ 0, %bb15.preheader ]
  %storemerge126 = add i64 %indvar111, 1
  %tmp117 = add i64 %indvar111, 2
  %scevgep119 = getelementptr [1025 x double]* @b, i64 0, i64 %storemerge126
  %scevgep118 = getelementptr [1025 x double]* @y, i64 0, i64 %storemerge126
  %28 = add i64 %indvar111, 1
  %29 = load double* %scevgep119, align 8
  %30 = icmp sgt i64 %storemerge126, 0
  br i1 %30, label %bb16.preheader, label %bb18

bb16.preheader:                                   ; preds = %bb15
  br label %bb16

bb16:                                             ; preds = %bb16.preheader, %bb16
  %31 = phi double [ %35, %bb16 ], [ %29, %bb16.preheader ]
  %storemerge423 = phi i64 [ %36, %bb16 ], [ 0, %bb16.preheader ]
  %scevgep114 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 %storemerge126, i64 %storemerge423
  %scevgep113 = getelementptr [1025 x double]* @y, i64 0, i64 %storemerge423
  %32 = load double* %scevgep114, align 8
  %33 = load double* %scevgep113, align 8
  %34 = fmul double %32, %33
  %35 = fsub double %31, %34
  %36 = add nsw i64 %storemerge423, 1
  %exitcond4 = icmp eq i64 %36, %storemerge126
  br i1 %exitcond4, label %bb18.loopexit, label %bb16

bb18.loopexit:                                    ; preds = %bb16
  %.lcssa = phi double [ %35, %bb16 ]
  br label %bb18

bb18:                                             ; preds = %bb18.loopexit, %bb15
  %w.tmp.032 = phi double [ %29, %bb15 ], [ %.lcssa, %bb18.loopexit ]
  store double %w.tmp.032, double* %scevgep118, align 8
  %37 = icmp sgt i64 %tmp117, %n
  br i1 %37, label %bb19.bb20_crit_edge, label %bb15

bb19.bb20_crit_edge:                              ; preds = %bb18
  %w.tmp.032.lcssa = phi double [ %w.tmp.032, %bb18 ]
  store double %w.tmp.032.lcssa, double* @w
  br label %bb20

bb20:                                             ; preds = %bb19.bb20_crit_edge, %bb14
  %38 = getelementptr inbounds [1025 x double]* @y, i64 0, i64 %n
  %39 = load double* %38, align 8
  %40 = getelementptr inbounds [1025 x [1025 x double]]* @a, i64 0, i64 %n, i64 %n
  %41 = load double* %40, align 8
  %42 = fdiv double %39, %41
  %43 = getelementptr inbounds [1025 x double]* @x, i64 0, i64 %n
  store double %42, double* %43, align 8
  %44 = add nsw i64 %n, -1
  %45 = icmp slt i64 %44, 0
  br i1 %45, label %return, label %bb.nph19

bb.nph19:                                         ; preds = %bb20
  %tmp86 = mul i64 %n, 1026
  %tmp90 = add i64 %n, 1
  %tmp94 = add i64 %tmp86, -1
  %tmp34 = add i64 %n, -1
  br label %bb21

bb21:                                             ; preds = %bb24, %bb.nph19
  %storemerge211 = phi i64 [ 0, %bb.nph19 ], [ %46, %bb24 ]
  %tmp23 = mul i64 %storemerge211, -1026
  %tmp24 = add i64 %tmp86, %tmp23
  %tmp27 = mul i64 %storemerge211, -1
  %tmp106 = add i64 %n, %tmp27
  %tmp31 = add i64 %tmp90, %tmp27
  %tmp109 = add i64 %storemerge211, 1
  %tmp35 = add i64 %tmp34, %tmp27
  %scevgep100 = getelementptr [1025 x double]* @y, i64 0, i64 %tmp35
  %scevgep99 = getelementptr [1025 x double]* @x, i64 0, i64 %tmp35
  %tmp38 = add i64 %tmp94, %tmp23
  %scevgep96 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 -1, i64 %tmp38
  %46 = add i64 %storemerge211, 1
  %47 = load double* %scevgep100, align 8
  %48 = icmp sgt i64 %tmp106, %n
  br i1 %48, label %bb24, label %bb22.preheader

bb22.preheader:                                   ; preds = %bb21
  br label %bb22

bb22:                                             ; preds = %bb22.preheader, %bb22
  %indvar = phi i64 [ %indvar.next, %bb22 ], [ 0, %bb22.preheader ]
  %w.tmp.0 = phi double [ %52, %bb22 ], [ %47, %bb22.preheader ]
  %tmp25 = add i64 %tmp24, %indvar
  %scevgep89 = getelementptr [1025 x [1025 x double]]* @a, i64 0, i64 -1, i64 %tmp25
  %tmp29 = add i64 %tmp106, %indvar
  %scevgep = getelementptr [1025 x double]* @x, i64 0, i64 %tmp29
  %tmp92 = add i64 %tmp31, %indvar
  %49 = load double* %scevgep89, align 8
  %50 = load double* %scevgep, align 8
  %51 = fmul double %49, %50
  %52 = fsub double %w.tmp.0, %51
  %53 = icmp sgt i64 %tmp92, %n
  %indvar.next = add i64 %indvar, 1
  br i1 %53, label %bb24.loopexit, label %bb22

bb24.loopexit:                                    ; preds = %bb22
  %.lcssa12 = phi double [ %52, %bb22 ]
  br label %bb24

bb24:                                             ; preds = %bb24.loopexit, %bb21
  %w.tmp.021 = phi double [ %47, %bb21 ], [ %.lcssa12, %bb24.loopexit ]
  %54 = load double* %scevgep96, align 8
  %55 = fdiv double %w.tmp.021, %54
  store double %55, double* %scevgep99, align 8
  %56 = icmp slt i64 %44, %tmp109
  br i1 %56, label %bb25.return_crit_edge, label %bb21

bb25.return_crit_edge:                            ; preds = %bb24
  %w.tmp.021.lcssa = phi double [ %w.tmp.021, %bb24 ]
  store double %w.tmp.021.lcssa, double* @w
  ret void

return:                                           ; preds = %bb20
  ret void
}
