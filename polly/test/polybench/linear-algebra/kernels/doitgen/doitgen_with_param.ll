; RUN: opt %loadPolly -correlated-propagation  %defaultOpts  -polly-cloog -analyze  %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [128 x [128 x [128 x double]]] zeroinitializer, align 32
@C4 = common global [128 x [128 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1
@sum = common global [128 x [128 x [128 x double]]] zeroinitializer, align 32

define void @scop_func(i64 %nr, i64 %nq, i64 %np) nounwind {
entry:
  %0 = icmp sgt i64 %nr, 0
  br i1 %0, label %bb.nph50, label %return

bb5.us:                                           ; preds = %bb3.us
  %.lcssa = phi double [ %5, %bb3.us ]
  store double %.lcssa, double* %scevgep54
  %1 = add nsw i64 %storemerge26.us, 1
  %exitcond = icmp eq i64 %1, %np
  br i1 %exitcond, label %bb9.loopexit, label %bb.nph.us

bb3.us:                                           ; preds = %bb.nph.us, %bb3.us
  %.tmp.0.us = phi double [ 0.000000e+00, %bb.nph.us ], [ %5, %bb3.us ]
  %storemerge45.us = phi i64 [ 0, %bb.nph.us ], [ %6, %bb3.us ]
  %scevgep = getelementptr [128 x [128 x double]]* @C4, i64 0, i64 %storemerge45.us, i64 %storemerge26.us
  %scevgep51 = getelementptr [128 x [128 x [128 x double]]]* @A, i64 0, i64 %storemerge43, i64 %storemerge113, i64 %storemerge45.us
  %2 = load double* %scevgep51, align 8
  %3 = load double* %scevgep, align 8
  %4 = fmul double %2, %3
  %5 = fadd double %.tmp.0.us, %4
  %6 = add nsw i64 %storemerge45.us, 1
  %exitcond1 = icmp eq i64 %6, %np
  br i1 %exitcond1, label %bb5.us, label %bb3.us

bb.nph.us:                                        ; preds = %bb.nph.us.preheader, %bb5.us
  %storemerge26.us = phi i64 [ %1, %bb5.us ], [ 0, %bb.nph.us.preheader ]
  %scevgep54 = getelementptr [128 x [128 x [128 x double]]]* @sum, i64 0, i64 %storemerge43, i64 %storemerge113, i64 %storemerge26.us
  store double 0.000000e+00, double* %scevgep54, align 8
  br label %bb3.us

bb8:                                              ; preds = %bb8.preheader, %bb8
  %storemerge311 = phi i64 [ %8, %bb8 ], [ 0, %bb8.preheader ]
  %scevgep62 = getelementptr [128 x [128 x [128 x double]]]* @sum, i64 0, i64 %storemerge43, i64 %storemerge113, i64 %storemerge311
  %scevgep61 = getelementptr [128 x [128 x [128 x double]]]* @A, i64 0, i64 %storemerge43, i64 %storemerge113, i64 %storemerge311
  %7 = load double* %scevgep62, align 8
  store double %7, double* %scevgep61, align 8
  %8 = add nsw i64 %storemerge311, 1
  %exitcond6 = icmp eq i64 %8, %np
  br i1 %exitcond6, label %bb10.loopexit, label %bb8

bb9.loopexit:                                     ; preds = %bb5.us
  br i1 %14, label %bb8.preheader, label %bb10

bb8.preheader:                                    ; preds = %bb9.loopexit
  br label %bb8

bb10.loopexit:                                    ; preds = %bb8
  br label %bb10

bb10:                                             ; preds = %bb10.loopexit, %bb6.preheader, %bb9.loopexit
  %storemerge12566 = phi i64 [ %storemerge113, %bb9.loopexit ], [ %storemerge113, %bb6.preheader ], [ %storemerge113, %bb10.loopexit ]
  %storemerge4464 = phi i64 [ %storemerge43, %bb9.loopexit ], [ %storemerge43, %bb6.preheader ], [ %storemerge43, %bb10.loopexit ]
  %9 = add nsw i64 %storemerge12566, 1
  %10 = icmp slt i64 %9, %nq
  br i1 %10, label %bb6.preheader.backedge, label %bb12

bb6.preheader.backedge:                           ; preds = %bb10, %bb12
  %storemerge43.be = phi i64 [ %storemerge4464, %bb10 ], [ %11, %bb12 ]
  %storemerge113.be = phi i64 [ %9, %bb10 ], [ 0, %bb12 ]
  br label %bb6.preheader

bb6.preheader:                                    ; preds = %bb6.preheader.backedge, %bb6.preheader.preheader
  %storemerge43 = phi i64 [ 0, %bb6.preheader.preheader ], [ %storemerge43.be, %bb6.preheader.backedge ]
  %storemerge113 = phi i64 [ 0, %bb6.preheader.preheader ], [ %storemerge113.be, %bb6.preheader.backedge ]
  br i1 %14, label %bb.nph.us.preheader, label %bb10

bb.nph.us.preheader:                              ; preds = %bb6.preheader
  br label %bb.nph.us

bb12:                                             ; preds = %bb10
  %11 = add nsw i64 %storemerge4464, 1
  %12 = icmp slt i64 %11, %nr
  br i1 %12, label %bb6.preheader.backedge, label %return.loopexit

bb.nph50:                                         ; preds = %entry
  %13 = icmp sgt i64 %nq, 0
  %14 = icmp sgt i64 %np, 0
  br i1 %13, label %bb6.preheader.preheader, label %return

bb6.preheader.preheader:                          ; preds = %bb.nph50
  br label %bb6.preheader

return.loopexit:                                  ; preds = %bb12
  br label %return

return:                                           ; preds = %return.loopexit, %bb.nph50, %entry
  ret void
}
; CHECK: for region: 'entry.split => return' in function 'scop_func':
