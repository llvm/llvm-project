; RUN: opt %loadPolly  %defaultOpts -polly-detect -analyze  %s | FileCheck %s
; region-simplify make polly fail to detect the canonical induction variable.
; XFAIL:*

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1024 x [1024 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb.nph28, label %return

bb1:                                              ; preds = %bb1.preheader, %bb1
  %indvar = phi i64 [ %indvar.next, %bb1 ], [ 0, %bb1.preheader ]
  %tmp27 = add i64 %tmp26, %indvar
  %scevgep = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 0, i64 %tmp27
  %1 = load double* %scevgep, align 8
  %2 = load double* %scevgep69, align 8
  %3 = fdiv double %1, %2
  store double %3, double* %scevgep, align 8
  %indvar.next = add i64 %indvar, 1
  %exitcond20 = icmp eq i64 %indvar.next, %tmp1
  br i1 %exitcond20, label %bb8.loopexit, label %bb1

bb5:                                              ; preds = %bb6.preheader, %bb5
  %indvar34 = phi i64 [ %indvar.next35, %bb5 ], [ 0, %bb6.preheader ]
  %tmp34 = add i64 %tmp26, %indvar34
  %scevgep45 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp32, i64 %tmp34
  %scevgep46 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 0, i64 %tmp34
  %4 = load double* %scevgep45, align 8
  %5 = load double* %scevgep55, align 8
  %6 = load double* %scevgep46, align 8
  %7 = fmul double %5, %6
  %8 = fsub double %4, %7
  store double %8, double* %scevgep45, align 8
  %indvar.next35 = add i64 %indvar34, 1
  %exitcond2 = icmp eq i64 %indvar.next35, %tmp1
  br i1 %exitcond2, label %bb8.loopexit4, label %bb5

bb8.loopexit:                                     ; preds = %bb1
  br i1 %10, label %bb6.preheader.preheader, label %bb9

bb6.preheader.preheader:                          ; preds = %bb8.loopexit
  br label %bb6.preheader

bb8.loopexit4:                                    ; preds = %bb5
  %exitcond11 = icmp eq i64 %tmp57, %tmp1
  br i1 %exitcond11, label %bb9.loopexit, label %bb6.preheader

bb6.preheader:                                    ; preds = %bb6.preheader.preheader, %bb8.loopexit4
  %indvar39 = phi i64 [ %tmp57, %bb8.loopexit4 ], [ 0, %bb6.preheader.preheader ]
  %tmp32 = add i64 %indvar39, 1
  %scevgep55 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp32, i64 %tmp25
  %tmp57 = add i64 %indvar39, 1
  br label %bb5

bb9.loopexit:                                     ; preds = %bb8.loopexit4
  br label %bb9

bb9:                                              ; preds = %bb9.loopexit, %bb2.preheader, %bb8.loopexit
  %exitcond = icmp eq i64 %9, %n
  br i1 %exitcond, label %return.loopexit, label %bb2.preheader

bb.nph28:                                         ; preds = %entry
  %tmp29 = add i64 %n, -1
  br label %bb2.preheader

bb2.preheader:                                    ; preds = %bb.nph28, %bb9
  %storemerge17 = phi i64 [ 0, %bb.nph28 ], [ %9, %bb9 ]
  %tmp25 = mul i64 %storemerge17, 1025
  %tmp26 = add i64 %tmp25, 1
  %tmp30 = mul i64 %storemerge17, -1
  %tmp1 = add i64 %tmp29, %tmp30
  %storemerge15 = add i64 %storemerge17, 1
  %scevgep69 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 0, i64 %tmp25
  %9 = add i64 %storemerge17, 1
  %10 = icmp slt i64 %storemerge15, %n
  br i1 %10, label %bb1.preheader, label %bb9

bb1.preheader:                                    ; preds = %bb2.preheader
  br label %bb1

return.loopexit:                                  ; preds = %bb9
  br label %return

return:                                           ; preds = %return.loopexit, %entry
  ret void
}
; CHECK: Valid Region for Scop: entry.split => return
