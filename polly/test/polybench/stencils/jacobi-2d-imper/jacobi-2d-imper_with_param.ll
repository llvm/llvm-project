; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
; region-simplify causes: Non canonical PHI node found
; XFAIL:*

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1024 x [1024 x double]] zeroinitializer, align 32
@B = common global [1024 x [1024 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %n) nounwind {
bb.nph35:
  %0 = add nsw i64 %n, -1
  %1 = icmp sgt i64 %0, 2
  %tmp = add i64 %n, -3
  br label %bb5.preheader

bb.nph:                                           ; preds = %bb.nph.preheader, %bb4
  %indvar36 = phi i64 [ %tmp50, %bb4 ], [ 0, %bb.nph.preheader ]
  %tmp13 = add i64 %indvar36, 1
  %tmp16 = add i64 %indvar36, 3
  %tmp18 = add i64 %indvar36, 2
  %scevgep40.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp18, i64 2
  %.pre = load double* %scevgep40.phi.trans.insert, align 16
  br label %bb2

bb2:                                              ; preds = %bb2, %bb.nph
  %2 = phi double [ %.pre, %bb.nph ], [ %5, %bb2 ]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp58, %bb2 ]
  %tmp14 = add i64 %indvar, 2
  %scevgep44 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp13, i64 %tmp14
  %scevgep42 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp16, i64 %tmp14
  %scevgep = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp18, i64 %tmp14
  %tmp20 = add i64 %indvar, 3
  %scevgep48 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp18, i64 %tmp20
  %tmp22 = add i64 %indvar, 1
  %scevgep46 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp18, i64 %tmp22
  %tmp58 = add i64 %indvar, 1
  %3 = load double* %scevgep46, align 8
  %4 = fadd double %2, %3
  %5 = load double* %scevgep48, align 8
  %6 = fadd double %4, %5
  %7 = load double* %scevgep42, align 8
  %8 = fadd double %6, %7
  %9 = load double* %scevgep44, align 8
  %10 = fadd double %8, %9
  %11 = fmul double %10, 2.000000e-01
  store double %11, double* %scevgep, align 8
  %exitcond1 = icmp eq i64 %tmp58, %tmp
  br i1 %exitcond1, label %bb4, label %bb2

bb4:                                              ; preds = %bb2
  %tmp50 = add i64 %indvar36, 1
  %exitcond = icmp eq i64 %tmp50, %tmp
  br i1 %exitcond, label %bb11.loopexit, label %bb.nph

bb8:                                              ; preds = %bb9.preheader, %bb8
  %indvar62 = phi i64 [ %indvar.next63, %bb8 ], [ 0, %bb9.preheader ]
  %tmp32 = add i64 %indvar62, 2
  %scevgep70 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp31, i64 %tmp32
  %scevgep69 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp31, i64 %tmp32
  %12 = load double* %scevgep70, align 8
  store double %12, double* %scevgep69, align 8
  %indvar.next63 = add i64 %indvar62, 1
  %exitcond25 = icmp eq i64 %indvar.next63, %tmp
  br i1 %exitcond25, label %bb10, label %bb8

bb10:                                             ; preds = %bb8
  %indvar.next66 = add i64 %indvar65, 1
  %exitcond30 = icmp eq i64 %indvar.next66, %tmp
  br i1 %exitcond30, label %bb12.loopexit, label %bb9.preheader

bb11.loopexit:                                    ; preds = %bb4
  br i1 %1, label %bb9.preheader.preheader, label %bb12

bb9.preheader.preheader:                          ; preds = %bb11.loopexit
  br label %bb9.preheader

bb9.preheader:                                    ; preds = %bb9.preheader.preheader, %bb10
  %indvar65 = phi i64 [ %indvar.next66, %bb10 ], [ 0, %bb9.preheader.preheader ]
  %tmp31 = add i64 %indvar65, 2
  br label %bb8

bb12.loopexit:                                    ; preds = %bb10
  br label %bb12

bb12:                                             ; preds = %bb12.loopexit, %bb5.preheader, %bb11.loopexit
  %13 = add nsw i64 %storemerge20, 1
  %exitcond35 = icmp eq i64 %13, 20
  br i1 %exitcond35, label %return, label %bb5.preheader

bb5.preheader:                                    ; preds = %bb12, %bb.nph35
  %storemerge20 = phi i64 [ 0, %bb.nph35 ], [ %13, %bb12 ]
  br i1 %1, label %bb.nph.preheader, label %bb12

bb.nph.preheader:                                 ; preds = %bb5.preheader
  br label %bb.nph

return:                                           ; preds = %bb12
  ret void
}
; CHECK: Valid Region for Scop: bb5.preheader => return

