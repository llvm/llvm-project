; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@X = common global [1024 x [1024 x double]] zeroinitializer, align 32
@A = common global [1024 x [1024 x double]] zeroinitializer, align 32
@B = common global [1024 x [1024 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func() nounwind {
bb.nph79:
  br label %bb5.preheader

bb.nph:                                           ; preds = %bb5.preheader, %bb4
  %storemerge112 = phi i64 [ %11, %bb4 ], [ 0, %bb5.preheader ]
  %scevgep83.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge112, i64 0
  %scevgep82.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge112, i64 0
  %.pre = load double* %scevgep82.phi.trans.insert, align 32
  %.pre143 = load double* %scevgep83.phi.trans.insert, align 32
  br label %bb2

bb2:                                              ; preds = %bb2, %bb.nph
  %0 = phi double [ %.pre143, %bb.nph ], [ %10, %bb2 ]
  %1 = phi double [ %.pre, %bb.nph ], [ %6, %bb2 ]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp87, %bb2 ]
  %tmp5 = add i64 %indvar, 1
  %scevgep81 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge112, i64 %tmp5
  %scevgep80 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %storemerge112, i64 %tmp5
  %scevgep = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge112, i64 %tmp5
  %tmp87 = add i64 %indvar, 1
  %2 = load double* %scevgep, align 8
  %3 = load double* %scevgep80, align 8
  %4 = fmul double %1, %3
  %5 = fdiv double %4, %0
  %6 = fsub double %2, %5
  store double %6, double* %scevgep, align 8
  %7 = load double* %scevgep81, align 8
  %8 = fmul double %3, %3
  %9 = fdiv double %8, %0
  %10 = fsub double %7, %9
  store double %10, double* %scevgep81, align 8
  %exitcond1 = icmp eq i64 %tmp87, 1023
  br i1 %exitcond1, label %bb4, label %bb2

bb4:                                              ; preds = %bb2
  %11 = add nsw i64 %storemerge112, 1
  %exitcond = icmp eq i64 %11, 1024
  br i1 %exitcond, label %bb7.loopexit, label %bb.nph

bb7.loopexit:                                     ; preds = %bb4
  br label %bb7

bb7:                                              ; preds = %bb7.loopexit, %bb7
  %storemerge217 = phi i64 [ %15, %bb7 ], [ 0, %bb7.loopexit ]
  %scevgep93 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge217, i64 1023
  %scevgep92 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge217, i64 1023
  %12 = load double* %scevgep92, align 8
  %13 = load double* %scevgep93, align 8
  %14 = fdiv double %12, %13
  store double %14, double* %scevgep92, align 8
  %15 = add nsw i64 %storemerge217, 1
  %exitcond11 = icmp eq i64 %15, 1024
  br i1 %exitcond11, label %bb12.preheader.loopexit, label %bb7

bb11:                                             ; preds = %bb12.preheader, %bb11
  %storemerge920 = phi i64 [ %23, %bb11 ], [ 0, %bb12.preheader ]
  %tmp22 = mul i64 %storemerge920, -1
  %tmp23 = add i64 %tmp22, 1021
  %scevgep100 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge323, i64 %tmp23
  %scevgep99 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %storemerge323, i64 %tmp23
  %scevgep98 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge323, i64 %tmp23
  %tmp27 = add i64 %tmp22, 1022
  %scevgep96 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge323, i64 %tmp27
  %16 = load double* %scevgep96, align 8
  %17 = load double* %scevgep98, align 8
  %18 = load double* %scevgep99, align 8
  %19 = fmul double %17, %18
  %20 = fsub double %16, %19
  %21 = load double* %scevgep100, align 8
  %22 = fdiv double %20, %21
  store double %22, double* %scevgep96, align 8
  %23 = add nsw i64 %storemerge920, 1
  %exitcond14 = icmp eq i64 %23, 1022
  br i1 %exitcond14, label %bb13, label %bb11

bb13:                                             ; preds = %bb11
  %24 = add nsw i64 %storemerge323, 1
  %exitcond21 = icmp eq i64 %24, 1024
  br i1 %exitcond21, label %bb18.preheader.loopexit, label %bb12.preheader

bb12.preheader.loopexit:                          ; preds = %bb7
  br label %bb12.preheader

bb12.preheader:                                   ; preds = %bb12.preheader.loopexit, %bb13
  %storemerge323 = phi i64 [ %24, %bb13 ], [ 0, %bb12.preheader.loopexit ]
  br label %bb11

bb17:                                             ; preds = %bb18.preheader, %bb17
  %storemerge828 = phi i64 [ %36, %bb17 ], [ 0, %bb18.preheader ]
  %scevgep114 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %indvar110, i64 %storemerge828
  %scevgep113 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %indvar110, i64 %storemerge828
  %scevgep116 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp38, i64 %storemerge828
  %scevgep115 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp38, i64 %storemerge828
  %scevgep112 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp38, i64 %storemerge828
  %25 = load double* %scevgep112, align 8
  %26 = load double* %scevgep113, align 8
  %27 = load double* %scevgep115, align 8
  %28 = fmul double %26, %27
  %29 = load double* %scevgep114, align 8
  %30 = fdiv double %28, %29
  %31 = fsub double %25, %30
  store double %31, double* %scevgep112, align 8
  %32 = load double* %scevgep116, align 8
  %33 = fmul double %27, %27
  %34 = fdiv double %33, %29
  %35 = fsub double %32, %34
  store double %35, double* %scevgep116, align 8
  %36 = add nsw i64 %storemerge828, 1
  %exitcond29 = icmp eq i64 %36, 1024
  br i1 %exitcond29, label %bb19, label %bb17

bb19:                                             ; preds = %bb17
  %tmp120 = add i64 %indvar110, 1
  %exitcond35 = icmp eq i64 %tmp120, 1023
  br i1 %exitcond35, label %bb22.loopexit, label %bb18.preheader

bb18.preheader.loopexit:                          ; preds = %bb13
  br label %bb18.preheader

bb18.preheader:                                   ; preds = %bb18.preheader.loopexit, %bb19
  %indvar110 = phi i64 [ %tmp120, %bb19 ], [ 0, %bb18.preheader.loopexit ]
  %tmp38 = add i64 %indvar110, 1
  br label %bb17

bb22.loopexit:                                    ; preds = %bb19
  br label %bb22

bb22:                                             ; preds = %bb22.loopexit, %bb22
  %storemerge535 = phi i64 [ %40, %bb22 ], [ 0, %bb22.loopexit ]
  %scevgep126 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 1023, i64 %storemerge535
  %scevgep125 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 1023, i64 %storemerge535
  %37 = load double* %scevgep125, align 8
  %38 = load double* %scevgep126, align 8
  %39 = fdiv double %37, %38
  store double %39, double* %scevgep125, align 8
  %40 = add nsw i64 %storemerge535, 1
  %exitcond42 = icmp eq i64 %40, 1024
  br i1 %exitcond42, label %bb27.preheader.loopexit, label %bb22

bb26:                                             ; preds = %bb27.preheader, %bb26
  %storemerge737 = phi i64 [ %48, %bb26 ], [ 0, %bb27.preheader ]
  %scevgep132 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp54, i64 %storemerge737
  %scevgep131 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp54, i64 %storemerge737
  %scevgep133 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp57, i64 %storemerge737
  %scevgep129 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp57, i64 %storemerge737
  %41 = load double* %scevgep129, align 8
  %42 = load double* %scevgep131, align 8
  %43 = load double* %scevgep132, align 8
  %44 = fmul double %42, %43
  %45 = fsub double %41, %44
  %46 = load double* %scevgep133, align 8
  %47 = fdiv double %45, %46
  store double %47, double* %scevgep129, align 8
  %48 = add nsw i64 %storemerge737, 1
  %exitcond45 = icmp eq i64 %48, 1024
  br i1 %exitcond45, label %bb28, label %bb26

bb28:                                             ; preds = %bb26
  %49 = add nsw i64 %storemerge639, 1
  %exitcond52 = icmp eq i64 %49, 1022
  br i1 %exitcond52, label %bb30, label %bb27.preheader

bb27.preheader.loopexit:                          ; preds = %bb22
  br label %bb27.preheader

bb27.preheader:                                   ; preds = %bb27.preheader.loopexit, %bb28
  %storemerge639 = phi i64 [ %49, %bb28 ], [ 0, %bb27.preheader.loopexit ]
  %tmp53 = mul i64 %storemerge639, -1
  %tmp54 = add i64 %tmp53, 1021
  %tmp57 = add i64 %tmp53, 1022
  br label %bb26

bb30:                                             ; preds = %bb28
  %50 = add nsw i64 %storemerge44, 1
  %exitcond60 = icmp eq i64 %50, 10
  br i1 %exitcond60, label %return, label %bb5.preheader

bb5.preheader:                                    ; preds = %bb30, %bb.nph79
  %storemerge44 = phi i64 [ 0, %bb.nph79 ], [ %50, %bb30 ]
  br label %bb.nph

return:                                           ; preds = %bb30
  ret void
}
; CHECK: Valid Region for Scop: bb5.preheader => return
