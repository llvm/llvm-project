; RUN: opt %loadPolly  %defaultOpts -polly-ast -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@r = common global [8000 x double] zeroinitializer, align 32
@p = common global [8000 x double] zeroinitializer, align 32
@A = common global [8000 x [8000 x double]] zeroinitializer, align 32
@s = common global [8000 x double] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@q = common global [8000 x double] zeroinitializer, align 32

define void @scop_func(i64 %nx, i64 %ny) nounwind {
entry:
  %0 = icmp sgt i64 %ny, 0
  br i1 %0, label %bb.preheader, label %bb7.preheader

bb.preheader:                                     ; preds = %entry
  br label %bb

bb:                                               ; preds = %bb.preheader, %bb
  %storemerge9 = phi i64 [ %1, %bb ], [ 0, %bb.preheader ]
  %scevgep20 = getelementptr [8000 x double]* @s, i64 0, i64 %storemerge9
  store double 0.000000e+00, double* %scevgep20, align 8
  %1 = add nsw i64 %storemerge9, 1
  %exitcond11 = icmp eq i64 %1, %ny
  br i1 %exitcond11, label %bb7.preheader.loopexit, label %bb

bb7.preheader.loopexit:                           ; preds = %bb
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb7.preheader.loopexit, %entry
  %2 = icmp sgt i64 %nx, 0
  br i1 %2, label %bb.nph8, label %return

bb.nph8:                                          ; preds = %bb7.preheader
  br i1 %0, label %bb.nph.us.preheader, label %bb6.preheader

bb.nph.us.preheader:                              ; preds = %bb.nph8
  br label %bb.nph.us

bb6.preheader:                                    ; preds = %bb.nph8
  br label %bb6

bb6.us:                                           ; preds = %bb4.us
  %.lcssa = phi double [ %10, %bb4.us ]
  store double %.lcssa, double* %scevgep15
  %3 = add nsw i64 %storemerge14.us, 1
  %exitcond = icmp eq i64 %3, %nx
  br i1 %exitcond, label %return.loopexit1, label %bb.nph.us

bb4.us:                                           ; preds = %bb.nph.us, %bb4.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %10, %bb4.us ]
  %storemerge23.us = phi i64 [ 0, %bb.nph.us ], [ %11, %bb4.us ]
  %scevgep11 = getelementptr [8000 x [8000 x double]]* @A, i64 0, i64 %storemerge14.us, i64 %storemerge23.us
  %scevgep12 = getelementptr [8000 x double]* @s, i64 0, i64 %storemerge23.us
  %scevgep = getelementptr [8000 x double]* @p, i64 0, i64 %storemerge23.us
  %4 = load double* %scevgep12, align 8
  %5 = load double* %scevgep11, align 8
  %6 = fmul double %12, %5
  %7 = fadd double %4, %6
  store double %7, double* %scevgep12, align 8
  %8 = load double* %scevgep, align 8
  %9 = fmul double %5, %8
  %10 = fadd double %.tmp.0.us, %9
  %11 = add nsw i64 %storemerge23.us, 1
  %exitcond4 = icmp eq i64 %11, %ny
  br i1 %exitcond4, label %bb6.us, label %bb4.us

bb.nph.us:                                        ; preds = %bb.nph.us.preheader, %bb6.us
  %storemerge14.us = phi i64 [ %3, %bb6.us ], [ 0, %bb.nph.us.preheader ]
  %scevgep16 = getelementptr [8000 x double]* @r, i64 0, i64 %storemerge14.us
  %scevgep15 = getelementptr [8000 x double]* @q, i64 0, i64 %storemerge14.us
  store double 0.000000e+00, double* %scevgep15, align 8
  %12 = load double* %scevgep16, align 8
  br label %bb4.us

bb6:                                              ; preds = %bb6.preheader, %bb6
  %indvar = phi i64 [ %indvar.next, %bb6 ], [ 0, %bb6.preheader ]
  %scevgep18 = getelementptr [8000 x double]* @q, i64 0, i64 %indvar
  store double 0.000000e+00, double* %scevgep18, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond2 = icmp eq i64 %indvar.next, %nx
  br i1 %exitcond2, label %return.loopexit, label %bb6

return.loopexit:                                  ; preds = %bb6
  br label %return

return.loopexit1:                                 ; preds = %bb6.us
  br label %return

return:                                           ; preds = %return.loopexit1, %return.loopexit, %bb7.preheader
  ret void
}
; CHECK:      for region: 'entry.split => return' in function 'scop_func':
; CHECK-NEXT: scop_func():

