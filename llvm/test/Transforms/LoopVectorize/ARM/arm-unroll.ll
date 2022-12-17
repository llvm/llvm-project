; RUN: opt < %s -passes=loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -S | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -mcpu=swift -S | FileCheck %s --check-prefix=SWIFT
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=1 -mtriple=thumbv7-apple-ios3.0.0 -mcpu=swift -S | FileCheck %s --check-prefix=SWIFTUNROLL

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

;CHECK-LABEL: @foo(
;CHECK: load <4 x i32>
;CHECK-NOT: load <4 x i32>
;CHECK: ret
;SWIFT-LABEL: @foo(
;SWIFT: load <4 x i32>
;SWIFT: load <4 x i32>
;SWIFT: ret
define i32 @foo(i32* nocapture %A, i32 %n) nounwind readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %i.02 = phi i32 [ %5, %.lr.ph ], [ 0, %0 ]
  %sum.01 = phi i32 [ %4, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, i32* %A, i32 %i.02
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, %sum.01
  %5 = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %5, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %4, %.lr.ph ]
  ret i32 %sum.0.lcssa
}

; Verify the register limit. On arm we don't have 16 allocatable registers.
;SWIFTUNROLL-LABEL: @register_limit(
;SWIFTUNROLL: load i32
;SWIFTUNROLL-NOT: load i32
define i32 @register_limit(i32* nocapture %A, i32 %n) {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:
  %i.02 = phi i32 [ %5, %.lr.ph ], [ 0, %0 ]
  %sum.01 = phi i32 [ %4, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %6, %.lr.ph ], [ 0, %0 ]
  %sum.03 = phi i32 [ %7, %.lr.ph ], [ 0, %0 ]
  %sum.04 = phi i32 [ %8, %.lr.ph ], [ 0, %0 ]
  %sum.05 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %sum.06 = phi i32 [ %10, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, i32* %A, i32 %i.02
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, %sum.01
  %5 = add nsw i32 %i.02, 1
  %6 = add nsw i32 %3, %sum.02
  %7 = add nsw i32 %3, %sum.03
  %8 = add nsw i32 %3, %sum.04
  %9 = add nsw i32 %3, %sum.05
  %10 = add nsw i32 %3, %sum.05
  %exitcond = icmp eq i32 %5, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %4, %.lr.ph ]
  %sum.1.lcssa = phi i32 [ 0, %0 ], [ %6, %.lr.ph ]
  %sum.2.lcssa = phi i32 [ 0, %0 ], [ %7, %.lr.ph ]
  %sum.4.lcssa = phi i32 [ 0, %0 ], [ %8, %.lr.ph ]
  %sum.5.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  %sum.6.lcssa = phi i32 [ 0, %0 ], [ %10, %.lr.ph ]
  ret i32 %sum.0.lcssa
}
