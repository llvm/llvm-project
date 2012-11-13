; RUN: opt %loadPolly  %defaultOpts -polly-cloog -analyze  %s| FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@alpha1 = common global double 0.000000e+00
@beta1 = common global double 0.000000e+00
@alpha2 = common global double 0.000000e+00
@beta2 = common global double 0.000000e+00
@A = common global [512 x [512 x double]] zeroinitializer, align 32
@B = common global [512 x [512 x double]] zeroinitializer, align 32
@C = common global [512 x [512 x double]] zeroinitializer, align 32
@D = common global [512 x [512 x double]] zeroinitializer, align 32
@E = common global [512 x [512 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %ni, i64 %nj, i64 %nk, i64 %nl) nounwind {
entry:
  %0 = icmp sgt i64 %ni, 0
  br i1 %0, label %bb.nph50, label %return

bb.nph35:                                         ; preds = %bb.nph35.preheader, %bb6
  %indvar17 = phi i64 [ 0, %bb.nph35.preheader ], [ %indvar.next18, %bb6 ]
  br i1 %8, label %bb.nph27.us.preheader, label %bb4.preheader

bb.nph27.us.preheader:                            ; preds = %bb.nph35
  br label %bb.nph27.us

bb4.preheader:                                    ; preds = %bb.nph35
  br label %bb4

bb4.us:                                           ; preds = %bb2.us
  %.lcssa20 = phi double [ %5, %bb2.us ]
  store double %.lcssa20, double* %scevgep64
  %1 = add nsw i64 %storemerge431.us, 1
  %exitcond24 = icmp eq i64 %1, %nj
  br i1 %exitcond24, label %bb6.loopexit2, label %bb.nph27.us

bb2.us:                                           ; preds = %bb.nph27.us, %bb2.us
  %.tmp.029.us = phi double [ 0.000000e+00, %bb.nph27.us ], [ %5, %bb2.us ]
  %storemerge526.us = phi i64 [ 0, %bb.nph27.us ], [ %6, %bb2.us ]
  %scevgep61 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %indvar17, i64 %storemerge526.us
  %scevgep60 = getelementptr [512 x [512 x double]]* @B, i64 0, i64 %storemerge526.us, i64 %storemerge431.us
  %2 = load double* %scevgep61, align 8
  %3 = load double* %scevgep60, align 8
  %4 = fmul double %2, %3
  %5 = fadd double %.tmp.029.us, %4
  %6 = add nsw i64 %storemerge526.us, 1
  %exitcond21 = icmp eq i64 %6, %nk
  br i1 %exitcond21, label %bb4.us, label %bb2.us

bb.nph27.us:                                      ; preds = %bb.nph27.us.preheader, %bb4.us
  %storemerge431.us = phi i64 [ %1, %bb4.us ], [ 0, %bb.nph27.us.preheader ]
  %scevgep64 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar17, i64 %storemerge431.us
  store double 0.000000e+00, double* %scevgep64, align 8
  br label %bb2.us

bb4:                                              ; preds = %bb4.preheader, %bb4
  %indvar67 = phi i64 [ %indvar.next68, %bb4 ], [ 0, %bb4.preheader ]
  %scevgep72 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar17, i64 %indvar67
  store double 0.000000e+00, double* %scevgep72, align 8
  %indvar.next68 = add i64 %indvar67, 1
  %exitcond16 = icmp eq i64 %indvar.next68, %nj
  br i1 %exitcond16, label %bb6.loopexit, label %bb4

bb6.loopexit:                                     ; preds = %bb4
  br label %bb6

bb6.loopexit2:                                    ; preds = %bb4.us
  br label %bb6

bb6:                                              ; preds = %bb6.loopexit2, %bb6.loopexit
  %indvar.next18 = add i64 %indvar17, 1
  %exitcond27 = icmp ne i64 %indvar.next18, %ni
  br i1 %exitcond27, label %bb.nph35, label %bb16.preheader.loopexit

bb.nph50:                                         ; preds = %entry
  %7 = icmp sgt i64 %nj, 0
  %8 = icmp sgt i64 %nk, 0
  br i1 %7, label %bb.nph35.preheader, label %bb16.preheader

bb.nph35.preheader:                               ; preds = %bb.nph50
  br label %bb.nph35

bb16.preheader.loopexit:                          ; preds = %bb6
  br label %bb16.preheader

bb16.preheader:                                   ; preds = %bb16.preheader.loopexit, %bb.nph50
  br i1 %0, label %bb.nph25, label %return

bb.nph11:                                         ; preds = %bb.nph11.preheader, %bb15
  %indvar4 = phi i64 [ 0, %bb.nph11.preheader ], [ %indvar.next5, %bb15 ]
  br i1 %16, label %bb.nph.us.preheader, label %bb13.preheader

bb.nph.us.preheader:                              ; preds = %bb.nph11
  br label %bb.nph.us

bb13.preheader:                                   ; preds = %bb.nph11
  br label %bb13

bb13.us:                                          ; preds = %bb11.us
  %.lcssa = phi double [ %13, %bb11.us ]
  store double %.lcssa, double* %scevgep54
  %9 = add nsw i64 %storemerge27.us, 1
  %exitcond = icmp eq i64 %9, %nl
  br i1 %exitcond, label %bb15.loopexit1, label %bb.nph.us

bb11.us:                                          ; preds = %bb.nph.us, %bb11.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %13, %bb11.us ]
  %storemerge36.us = phi i64 [ 0, %bb.nph.us ], [ %14, %bb11.us ]
  %scevgep51 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar4, i64 %storemerge36.us
  %scevgep = getelementptr [512 x [512 x double]]* @D, i64 0, i64 %storemerge36.us, i64 %storemerge27.us
  %10 = load double* %scevgep51, align 8
  %11 = load double* %scevgep, align 8
  %12 = fmul double %10, %11
  %13 = fadd double %.tmp.0.us, %12
  %14 = add nsw i64 %storemerge36.us, 1
  %exitcond7 = icmp eq i64 %14, %nj
  br i1 %exitcond7, label %bb13.us, label %bb11.us

bb.nph.us:                                        ; preds = %bb.nph.us.preheader, %bb13.us
  %storemerge27.us = phi i64 [ %9, %bb13.us ], [ 0, %bb.nph.us.preheader ]
  %scevgep54 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %indvar4, i64 %storemerge27.us
  store double 0.000000e+00, double* %scevgep54, align 8
  br label %bb11.us

bb13:                                             ; preds = %bb13.preheader, %bb13
  %indvar = phi i64 [ %indvar.next, %bb13 ], [ 0, %bb13.preheader ]
  %scevgep57 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %indvar4, i64 %indvar
  store double 0.000000e+00, double* %scevgep57, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond3 = icmp eq i64 %indvar.next, %nl
  br i1 %exitcond3, label %bb15.loopexit, label %bb13

bb15.loopexit:                                    ; preds = %bb13
  br label %bb15

bb15.loopexit1:                                   ; preds = %bb13.us
  br label %bb15

bb15:                                             ; preds = %bb15.loopexit1, %bb15.loopexit
  %indvar.next5 = add i64 %indvar4, 1
  %exitcond12 = icmp ne i64 %indvar.next5, %ni
  br i1 %exitcond12, label %bb.nph11, label %return.loopexit

bb.nph25:                                         ; preds = %bb16.preheader
  %15 = icmp sgt i64 %nl, 0
  %16 = icmp sgt i64 %nj, 0
  br i1 %15, label %bb.nph11.preheader, label %return

bb.nph11.preheader:                               ; preds = %bb.nph25
  br label %bb.nph11

return.loopexit:                                  ; preds = %bb15
  br label %return

return:                                           ; preds = %return.loopexit, %bb.nph25, %bb16.preheader, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
