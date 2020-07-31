; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Previously in this case the max backedge count would be computed as 1/0, which
; is correct but undesirable.  It would also not fold as a constant, tripping
; asserts in SCEV.

define void @pluto(i32 %arg) {
; CHECK-LABEL: Classifying expressions for: @pluto
; CHECK: Loop %bb2: max backedge-taken count is 2
bb:
  %tmp = ashr i32 %arg, 31
  %tmp1 = add nsw i32 %tmp, 2
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp4, %bb2 ]
  %tmp4 = add nuw nsw i32 %tmp1, %tmp3
  %tmp5 = icmp ult i32 %tmp4, 2
  br i1 %tmp5, label %bb2, label %bb6

bb6:                                              ; preds = %bb2
  ret void
}
