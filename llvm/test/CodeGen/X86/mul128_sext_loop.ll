; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

define void @test(i64* nocapture %arr, i64 %arrsize, i64 %factor) nounwind uwtable {
  %1 = icmp sgt i64 %arrsize, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = sext i64 %factor to i128
  br label %3

; <label>:3                                       ; preds = %3, %.lr.ph
; CHECK-NOT: mul
; CHECK: imulq
; CHECK-NOT: mul
  %carry.02 = phi i128 [ 0, %.lr.ph ], [ %10, %3 ]
  %i.01 = phi i64 [ 0, %.lr.ph ], [ %11, %3 ]
  %4 = getelementptr inbounds i64, i64* %arr, i64 %i.01
  %5 = load i64, i64* %4, align 8
  %6 = sext i64 %5 to i128
  %7 = mul nsw i128 %6, %2
  %8 = add nsw i128 %7, %carry.02
  %.tr = trunc i128 %8 to i64
  %9 = and i64 %.tr, 9223372036854775807
  store i64 %9, i64* %4, align 8
  %10 = ashr i128 %8, 63
  %11 = add nsw i64 %i.01, 1
  %exitcond = icmp eq i64 %11, %arrsize
  br i1 %exitcond, label %._crit_edge, label %3

._crit_edge:                                      ; preds = %3, %0
  ret void
}
