; RUN: opt %loadPolly  %defaultOpts -polly-detect -polly-cloog -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@alpha = common global double 0.000000e+00
@beta = common global double 0.000000e+00
@A = common global [512 x [512 x double]] zeroinitializer, align 32
@B = common global [512 x [512 x double]] zeroinitializer, align 32
@C = common global [512 x [512 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %ni, i64 %nj, i64 %nk) nounwind {
entry:
  %0 = icmp sgt i64 %ni, 0
  br i1 %0, label %bb.nph26, label %return

bb.nph8:                                          ; preds = %bb.nph8.preheader, %bb6
  %indvar3 = phi i64 [ 0, %bb.nph8.preheader ], [ %indvar.next4, %bb6 ]
  br i1 %14, label %bb.nph.us.preheader, label %bb4.preheader

bb.nph.us.preheader:                              ; preds = %bb.nph8
  br label %bb.nph.us

bb4.preheader:                                    ; preds = %bb.nph8
  br label %bb4

bb4.us:                                           ; preds = %bb2.us
  %.lcssa = phi double [ %6, %bb2.us ]
  store double %.lcssa, double* %scevgep30
  %1 = add nsw i64 %storemerge14.us, 1
  %exitcond = icmp eq i64 %1, %nj
  br i1 %exitcond, label %bb6.loopexit1, label %bb.nph.us

bb2.us:                                           ; preds = %bb.nph.us, %bb2.us
  %.tmp.0.us = phi double [ %9, %bb.nph.us ], [ %6, %bb2.us ]
  %storemerge23.us = phi i64 [ 0, %bb.nph.us ], [ %7, %bb2.us ]
  %scevgep27 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %indvar3, i64 %storemerge23.us
  %scevgep = getelementptr [512 x [512 x double]]* @B, i64 0, i64 %storemerge23.us, i64 %storemerge14.us
  %2 = load double* %scevgep27, align 8
  %3 = fmul double %2, %15
  %4 = load double* %scevgep, align 8
  %5 = fmul double %3, %4
  %6 = fadd double %.tmp.0.us, %5
  %7 = add nsw i64 %storemerge23.us, 1
  %exitcond6 = icmp eq i64 %7, %nk
  br i1 %exitcond6, label %bb4.us, label %bb2.us

bb.nph.us:                                        ; preds = %bb.nph.us.preheader, %bb4.us
  %storemerge14.us = phi i64 [ %1, %bb4.us ], [ 0, %bb.nph.us.preheader ]
  %scevgep30 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar3, i64 %storemerge14.us
  %8 = load double* %scevgep30, align 8
  %9 = fmul double %8, %13
  store double %9, double* %scevgep30, align 8
  br label %bb2.us

bb4:                                              ; preds = %bb4.preheader, %bb4
  %indvar = phi i64 [ %indvar.next, %bb4 ], [ 0, %bb4.preheader ]
  %scevgep35 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar3, i64 %indvar
  %10 = load double* %scevgep35, align 8
  %11 = fmul double %10, %13
  store double %11, double* %scevgep35, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond2 = icmp eq i64 %indvar.next, %nj
  br i1 %exitcond2, label %bb6.loopexit, label %bb4

bb6.loopexit:                                     ; preds = %bb4
  br label %bb6

bb6.loopexit1:                                    ; preds = %bb4.us
  br label %bb6

bb6:                                              ; preds = %bb6.loopexit1, %bb6.loopexit
  %indvar.next4 = add i64 %indvar3, 1
  %exitcond11 = icmp ne i64 %indvar.next4, %ni
  br i1 %exitcond11, label %bb.nph8, label %return.loopexit

bb.nph26:                                         ; preds = %entry
  %12 = icmp sgt i64 %nj, 0
  %13 = load double* @beta, align 8
  %14 = icmp sgt i64 %nk, 0
  %15 = load double* @alpha, align 8
  br i1 %12, label %bb.nph8.preheader, label %return

bb.nph8.preheader:                                ; preds = %bb.nph26
  br label %bb.nph8

return.loopexit:                                  ; preds = %bb6
  br label %return

return:                                           ; preds = %return.loopexit, %bb.nph26, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
