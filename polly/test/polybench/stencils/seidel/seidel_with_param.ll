; RUN: opt %loadPolly  %defaultOpts -polly-analyze-ir  -print-top-scop-only -analyze %s | FileCheck %s
; XFAIL: *
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1024 x [1024 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %n) nounwind {
bb.nph20:
  %0 = add nsw i64 %n, -2
  %1 = icmp slt i64 %0, 1
  br i1 %1, label %return, label %bb.nph8.preheader

bb.nph8.preheader:                                ; preds = %bb.nph20
  br label %bb.nph8

bb.nph:                                           ; preds = %bb.nph.preheader, %bb4
  %indvar21 = phi i64 [ %tmp39, %bb4 ], [ 0, %bb.nph.preheader ]
  %tmp5 = add i64 %indvar21, 1
  %tmp43 = add i64 %indvar21, 2
  %scevgep26.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %indvar21, i64 1
  %scevgep.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp43, i64 1
  %scevgep30.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp5, i64 0
  %scevgep25.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp5, i64 1
  %tmp39 = add i64 %indvar21, 1
  %.pre = load double* %scevgep26.phi.trans.insert, align 8
  %.pre47 = load double* %scevgep25.phi.trans.insert, align 8
  %.pre48 = load double* %scevgep.phi.trans.insert, align 8
  %.pre49 = load double* %scevgep30.phi.trans.insert, align 32
  br label %bb2

bb2:                                              ; preds = %bb2, %bb.nph
  %2 = phi double [ %.pre49, %bb.nph ], [ %19, %bb2 ]
  %3 = phi double [ %.pre48, %bb.nph ], [ %17, %bb2 ]
  %4 = phi double [ %.pre47, %bb.nph ], [ %12, %bb2 ]
  %5 = phi double [ %.pre, %bb.nph ], [ %8, %bb2 ]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp37, %bb2 ]
  %tmp4 = add i64 %indvar, 2
  %scevgep29 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %indvar21, i64 %tmp4
  %scevgep27 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %indvar21, i64 %indvar
  %scevgep31 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp5, i64 %tmp4
  %tmp6 = add i64 %indvar, 1
  %scevgep25 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp5, i64 %tmp6
  %scevgep33 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp43, i64 %tmp4
  %scevgep32 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp43, i64 %indvar
  %tmp34 = add i64 %indvar, 2
  %tmp37 = add i64 %indvar, 1
  %6 = load double* %scevgep27, align 8
  %7 = fadd double %6, %5
  %8 = load double* %scevgep29, align 8
  %9 = fadd double %7, %8
  %10 = fadd double %9, %2
  %11 = fadd double %10, %4
  %12 = load double* %scevgep31, align 8
  %13 = fadd double %11, %12
  %14 = load double* %scevgep32, align 8
  %15 = fadd double %13, %14
  %16 = fadd double %15, %3
  %17 = load double* %scevgep33, align 8
  %18 = fadd double %16, %17
  %19 = fdiv double %18, 9.000000e+00
  store double %19, double* %scevgep25, align 8
  %20 = icmp slt i64 %0, %tmp34
  br i1 %20, label %bb4, label %bb2

bb4:                                              ; preds = %bb2
  %21 = icmp slt i64 %0, %tmp43
  br i1 %21, label %bb6.loopexit, label %bb.nph

bb.nph8:                                          ; preds = %bb.nph8.preheader, %bb6
  %storemerge9 = phi i64 [ %22, %bb6 ], [ 0, %bb.nph8.preheader ]
  br i1 %1, label %bb6, label %bb.nph.preheader

bb.nph.preheader:                                 ; preds = %bb.nph8
  br label %bb.nph

bb6.loopexit:                                     ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb6.loopexit, %bb.nph8
  %22 = add nsw i64 %storemerge9, 1
  %exitcond8 = icmp eq i64 %22, 20
  br i1 %exitcond8, label %return.loopexit, label %bb.nph8

return.loopexit:                                  ; preds = %bb6
  br label %return

return:                                           ; preds = %return.loopexit, %bb.nph20
  ret void
}
