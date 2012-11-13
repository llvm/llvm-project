; RUN: opt %loadPolly  %defaultOpts -polly-cloog -analyze  %s| FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [512 x [512 x double]] zeroinitializer, align 32
@B = common global [512 x [512 x double]] zeroinitializer, align 32
@C = common global [512 x [512 x double]] zeroinitializer, align 32
@D = common global [512 x [512 x double]] zeroinitializer, align 32
@E = common global [512 x [512 x double]] zeroinitializer, align 32
@F = common global [512 x [512 x double]] zeroinitializer, align 32
@G = common global [512 x [512 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %ni, i64 %nj, i64 %nk, i64 %nl, i64 %nm) nounwind {
entry:
  %0 = icmp sgt i64 %ni, 0
  br i1 %0, label %bb.nph76.bb.nph76.split_crit_edge, label %return

bb.nph62:                                         ; preds = %bb.nph76.bb.nph76.split_crit_edge, %bb6
  %indvar33 = phi i64 [ 0, %bb.nph76.bb.nph76.split_crit_edge ], [ %indvar.next34, %bb6 ]
  br i1 %7, label %bb.nph54.us.preheader, label %bb4.preheader

bb.nph54.us.preheader:                            ; preds = %bb.nph62
  br label %bb.nph54.us

bb4.preheader:                                    ; preds = %bb.nph62
  br label %bb4

bb4.us:                                           ; preds = %bb2.us
  %.lcssa36 = phi double [ %5, %bb2.us ]
  store double %.lcssa36, double* %scevgep105
  %1 = add nsw i64 %storemerge758.us, 1
  %exitcond40 = icmp eq i64 %1, %ni
  br i1 %exitcond40, label %bb6.loopexit3, label %bb.nph54.us

bb2.us:                                           ; preds = %bb.nph54.us, %bb2.us
  %.tmp.056.us = phi double [ 0.000000e+00, %bb.nph54.us ], [ %5, %bb2.us ]
  %storemerge853.us = phi i64 [ 0, %bb.nph54.us ], [ %6, %bb2.us ]
  %scevgep102 = getelementptr [512 x [512 x double]]* @A, i64 0, i64 %indvar33, i64 %storemerge853.us
  %scevgep101 = getelementptr [512 x [512 x double]]* @B, i64 0, i64 %storemerge853.us, i64 %storemerge758.us
  %2 = load double* %scevgep102, align 8
  %3 = load double* %scevgep101, align 8
  %4 = fmul double %2, %3
  %5 = fadd double %.tmp.056.us, %4
  %6 = add nsw i64 %storemerge853.us, 1
  %exitcond37 = icmp eq i64 %6, %nk
  br i1 %exitcond37, label %bb4.us, label %bb2.us

bb.nph54.us:                                      ; preds = %bb.nph54.us.preheader, %bb4.us
  %storemerge758.us = phi i64 [ %1, %bb4.us ], [ 0, %bb.nph54.us.preheader ]
  %scevgep105 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %indvar33, i64 %storemerge758.us
  store double 0.000000e+00, double* %scevgep105, align 8
  br label %bb2.us

bb4:                                              ; preds = %bb4.preheader, %bb4
  %indvar108 = phi i64 [ %indvar.next109, %bb4 ], [ 0, %bb4.preheader ]
  %scevgep113 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %indvar33, i64 %indvar108
  store double 0.000000e+00, double* %scevgep113, align 8
  %indvar.next109 = add i64 %indvar108, 1
  %exitcond32 = icmp eq i64 %indvar.next109, %ni
  br i1 %exitcond32, label %bb6.loopexit, label %bb4

bb6.loopexit:                                     ; preds = %bb4
  br label %bb6

bb6.loopexit3:                                    ; preds = %bb4.us
  br label %bb6

bb6:                                              ; preds = %bb6.loopexit3, %bb6.loopexit
  %indvar.next34 = add i64 %indvar33, 1
  %exitcond43 = icmp ne i64 %indvar.next34, %ni
  br i1 %exitcond43, label %bb.nph62, label %bb16.preheader

bb.nph76.bb.nph76.split_crit_edge:                ; preds = %entry
  %7 = icmp sgt i64 %nk, 0
  br label %bb.nph62

bb16.preheader:                                   ; preds = %bb6
  br i1 %0, label %bb.nph52.bb.nph52.split_crit_edge, label %return

bb.nph38:                                         ; preds = %bb.nph52.bb.nph52.split_crit_edge, %bb15
  %indvar18 = phi i64 [ 0, %bb.nph52.bb.nph52.split_crit_edge ], [ %indvar.next19, %bb15 ]
  br i1 %14, label %bb.nph30.us.preheader, label %bb13.preheader

bb.nph30.us.preheader:                            ; preds = %bb.nph38
  br label %bb.nph30.us

bb13.preheader:                                   ; preds = %bb.nph38
  br label %bb13

bb13.us:                                          ; preds = %bb11.us
  %.lcssa21 = phi double [ %12, %bb11.us ]
  store double %.lcssa21, double* %scevgep90
  %8 = add nsw i64 %storemerge534.us, 1
  %exitcond25 = icmp eq i64 %8, %ni
  br i1 %exitcond25, label %bb15.loopexit2, label %bb.nph30.us

bb11.us:                                          ; preds = %bb.nph30.us, %bb11.us
  %.tmp.032.us = phi double [ 0.000000e+00, %bb.nph30.us ], [ %12, %bb11.us ]
  %storemerge629.us = phi i64 [ 0, %bb.nph30.us ], [ %13, %bb11.us ]
  %scevgep87 = getelementptr [512 x [512 x double]]* @C, i64 0, i64 %indvar18, i64 %storemerge629.us
  %scevgep86 = getelementptr [512 x [512 x double]]* @D, i64 0, i64 %storemerge629.us, i64 %storemerge534.us
  %9 = load double* %scevgep87, align 8
  %10 = load double* %scevgep86, align 8
  %11 = fmul double %9, %10
  %12 = fadd double %.tmp.032.us, %11
  %13 = add nsw i64 %storemerge629.us, 1
  %exitcond22 = icmp eq i64 %13, %nk
  br i1 %exitcond22, label %bb13.us, label %bb11.us

bb.nph30.us:                                      ; preds = %bb.nph30.us.preheader, %bb13.us
  %storemerge534.us = phi i64 [ %8, %bb13.us ], [ 0, %bb.nph30.us.preheader ]
  %scevgep90 = getelementptr [512 x [512 x double]]* @F, i64 0, i64 %indvar18, i64 %storemerge534.us
  store double 0.000000e+00, double* %scevgep90, align 8
  br label %bb11.us

bb13:                                             ; preds = %bb13.preheader, %bb13
  %indvar93 = phi i64 [ %indvar.next94, %bb13 ], [ 0, %bb13.preheader ]
  %scevgep98 = getelementptr [512 x [512 x double]]* @F, i64 0, i64 %indvar18, i64 %indvar93
  store double 0.000000e+00, double* %scevgep98, align 8
  %indvar.next94 = add i64 %indvar93, 1
  %exitcond17 = icmp eq i64 %indvar.next94, %ni
  br i1 %exitcond17, label %bb15.loopexit, label %bb13

bb15.loopexit:                                    ; preds = %bb13
  br label %bb15

bb15.loopexit2:                                   ; preds = %bb13.us
  br label %bb15

bb15:                                             ; preds = %bb15.loopexit2, %bb15.loopexit
  %indvar.next19 = add i64 %indvar18, 1
  %exitcond28 = icmp ne i64 %indvar.next19, %ni
  br i1 %exitcond28, label %bb.nph38, label %bb25.preheader

bb.nph52.bb.nph52.split_crit_edge:                ; preds = %bb16.preheader
  %14 = icmp sgt i64 %nk, 0
  br label %bb.nph38

bb25.preheader:                                   ; preds = %bb15
  br i1 %0, label %bb.nph28.bb.nph28.split_crit_edge, label %return

bb.nph14:                                         ; preds = %bb.nph28.bb.nph28.split_crit_edge, %bb24
  %indvar5 = phi i64 [ 0, %bb.nph28.bb.nph28.split_crit_edge ], [ %indvar.next6, %bb24 ]
  br i1 %21, label %bb.nph.us.preheader, label %bb22.preheader

bb.nph.us.preheader:                              ; preds = %bb.nph14
  br label %bb.nph.us

bb22.preheader:                                   ; preds = %bb.nph14
  br label %bb22

bb22.us:                                          ; preds = %bb20.us
  %.lcssa = phi double [ %19, %bb20.us ]
  store double %.lcssa, double* %scevgep80
  %15 = add nsw i64 %storemerge310.us, 1
  %exitcond = icmp eq i64 %15, %ni
  br i1 %exitcond, label %bb24.loopexit1, label %bb.nph.us

bb20.us:                                          ; preds = %bb.nph.us, %bb20.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %19, %bb20.us ]
  %storemerge49.us = phi i64 [ 0, %bb.nph.us ], [ %20, %bb20.us ]
  %scevgep77 = getelementptr [512 x [512 x double]]* @E, i64 0, i64 %indvar5, i64 %storemerge49.us
  %scevgep = getelementptr [512 x [512 x double]]* @F, i64 0, i64 %storemerge49.us, i64 %storemerge310.us
  %16 = load double* %scevgep77, align 8
  %17 = load double* %scevgep, align 8
  %18 = fmul double %16, %17
  %19 = fadd double %.tmp.0.us, %18
  %20 = add nsw i64 %storemerge49.us, 1
  %exitcond8 = icmp eq i64 %20, %nk
  br i1 %exitcond8, label %bb22.us, label %bb20.us

bb.nph.us:                                        ; preds = %bb.nph.us.preheader, %bb22.us
  %storemerge310.us = phi i64 [ %15, %bb22.us ], [ 0, %bb.nph.us.preheader ]
  %scevgep80 = getelementptr [512 x [512 x double]]* @G, i64 0, i64 %indvar5, i64 %storemerge310.us
  store double 0.000000e+00, double* %scevgep80, align 8
  br label %bb20.us

bb22:                                             ; preds = %bb22.preheader, %bb22
  %indvar = phi i64 [ %indvar.next, %bb22 ], [ 0, %bb22.preheader ]
  %scevgep83 = getelementptr [512 x [512 x double]]* @G, i64 0, i64 %indvar5, i64 %indvar
  store double 0.000000e+00, double* %scevgep83, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond4 = icmp eq i64 %indvar.next, %ni
  br i1 %exitcond4, label %bb24.loopexit, label %bb22

bb24.loopexit:                                    ; preds = %bb22
  br label %bb24

bb24.loopexit1:                                   ; preds = %bb22.us
  br label %bb24

bb24:                                             ; preds = %bb24.loopexit1, %bb24.loopexit
  %indvar.next6 = add i64 %indvar5, 1
  %exitcond13 = icmp ne i64 %indvar.next6, %ni
  br i1 %exitcond13, label %bb.nph14, label %return.loopexit

bb.nph28.bb.nph28.split_crit_edge:                ; preds = %bb25.preheader
  %21 = icmp sgt i64 %nk, 0
  br label %bb.nph14

return.loopexit:                                  ; preds = %bb24
  br label %return

return:                                           ; preds = %return.loopexit, %bb25.preheader, %bb16.preheader, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
