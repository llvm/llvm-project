; RUN: opt %loadPolly  %defaultOpts -polly-ast -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@x = common global [8000 x double] zeroinitializer, align 32
@A = common global [8000 x [8000 x double]] zeroinitializer, align 32
@y = common global [8000 x double] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@tmp = common global [8000 x double] zeroinitializer, align 32

define void @scop_func(i64 %nx, i64 %ny) nounwind {
entry:
  %0 = icmp sgt i64 %nx, 0
  br i1 %0, label %bb.preheader, label %bb10.preheader

bb.preheader:                                     ; preds = %entry
  br label %bb

bb:                                               ; preds = %bb.preheader, %bb
  %storemerge15 = phi i64 [ %1, %bb ], [ 0, %bb.preheader ]
  %scevgep26 = getelementptr [8000 x double]* @y, i64 0, i64 %storemerge15
  store double 0.000000e+00, double* %scevgep26, align 8
  %1 = add nsw i64 %storemerge15, 1
  %exitcond10 = icmp eq i64 %1, %nx
  br i1 %exitcond10, label %bb10.preheader.loopexit, label %bb

bb10.preheader.loopexit:                          ; preds = %bb
  br label %bb10.preheader

bb10.preheader:                                   ; preds = %bb10.preheader.loopexit, %entry
  %2 = icmp sgt i64 %ny, 0
  br i1 %2, label %bb.nph.preheader, label %return

bb.nph.preheader:                                 ; preds = %bb10.preheader
  br label %bb.nph

bb.nph:                                           ; preds = %bb.nph.preheader, %bb9
  %storemerge17 = phi i64 [ %13, %bb9 ], [ 0, %bb.nph.preheader ]
  %scevgep24 = getelementptr [8000 x double]* @tmp, i64 0, i64 %storemerge17
  store double 0.000000e+00, double* %scevgep24, align 8
  br label %bb4

bb4:                                              ; preds = %bb4, %bb.nph
  %.tmp.0 = phi double [ 0.000000e+00, %bb.nph ], [ %6, %bb4 ]
  %storemerge24 = phi i64 [ 0, %bb.nph ], [ %7, %bb4 ]
  %scevgep17 = getelementptr [8000 x [8000 x double]]* @A, i64 0, i64 %storemerge17, i64 %storemerge24
  %scevgep = getelementptr [8000 x double]* @x, i64 0, i64 %storemerge24
  %3 = load double* %scevgep17, align 8
  %4 = load double* %scevgep, align 8
  %5 = fmul double %3, %4
  %6 = fadd double %.tmp.0, %5
  %7 = add nsw i64 %storemerge24, 1
  %exitcond1 = icmp eq i64 %7, %ny
  br i1 %exitcond1, label %bb8.loopexit, label %bb4

bb7:                                              ; preds = %bb8.loopexit, %bb7
  %storemerge35 = phi i64 [ %12, %bb7 ], [ 0, %bb8.loopexit ]
  %scevgep19 = getelementptr [8000 x [8000 x double]]* @A, i64 0, i64 %storemerge17, i64 %storemerge35
  %scevgep20 = getelementptr [8000 x double]* @y, i64 0, i64 %storemerge35
  %8 = load double* %scevgep20, align 8
  %9 = load double* %scevgep19, align 8
  %10 = fmul double %9, %.lcssa
  %11 = fadd double %8, %10
  store double %11, double* %scevgep20, align 8
  %12 = add nsw i64 %storemerge35, 1
  %exitcond = icmp eq i64 %12, %ny
  br i1 %exitcond, label %bb9, label %bb7

bb8.loopexit:                                     ; preds = %bb4
  %.lcssa = phi double [ %6, %bb4 ]
  store double %.lcssa, double* %scevgep24
  br label %bb7

bb9:                                              ; preds = %bb7
  %13 = add nsw i64 %storemerge17, 1
  %exitcond6 = icmp eq i64 %13, %ny
  br i1 %exitcond6, label %return.loopexit, label %bb.nph

return.loopexit:                                  ; preds = %bb9
  br label %return

return:                                           ; preds = %return.loopexit, %bb10.preheader
  ret void
}
; CHECK:       for region: 'entry.split => return' in function 'scop_func':
; CHECK-NEXT: scop_func():
