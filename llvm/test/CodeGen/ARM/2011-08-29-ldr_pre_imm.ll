; RUN: llc -O3 -mtriple=armv6-apple-darwin -relocation-model=pic < %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:64-n32"

define void @compdecomp() nounwind {
entry:
  %heap = alloca [256 x i32], align 4
  br i1 undef, label %bb25.lr.ph, label %bb17

bb17:                                             ; preds = %bb17, %entry
  br label %bb17

bb25.lr.ph:                                       ; preds = %entry
  %0 = sdiv i32 undef, 2
  br label %bb5.i

bb.i:                                             ; preds = %bb5.i
  %1 = shl nsw i32 %k_addr.0.i, 1
  %.sum8.i = add i32 %1, -1
  %2 = getelementptr inbounds [256 x i32]* %heap, i32 0, i32 %.sum8.i
  %3 = load i32* %2, align 4
  br i1 false, label %bb5.i, label %bb4.i

bb4.i:                                            ; preds = %bb.i
  %.sum10.i = add i32 %k_addr.0.i, -1
  %4 = getelementptr inbounds [256 x i32]* %heap, i32 0, i32 %.sum10.i
  store i32 %3, i32* %4, align 4
  br label %bb5.i

bb5.i:                                            ; preds = %bb5.i, %bb4.i, %bb.i, %bb25.lr.ph
  %k_addr.0.i = phi i32 [ %1, %bb4.i ], [ undef, %bb25.lr.ph ], [ undef, %bb5.i ], [ undef, %bb.i ]
  %5 = icmp slt i32 %0, %k_addr.0.i
  br i1 %5, label %bb5.i, label %bb.i
}
