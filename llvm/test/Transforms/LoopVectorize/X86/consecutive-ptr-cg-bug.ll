; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; PR34965/D39346

; LV retains the original scalar loop intact as remainder loop. However,
; after this transformation, analysis information concerning the remainder
; loop may differ from the original scalar loop. This test is an example of
; that behaviour, where values inside the remainder loop which SCEV could
; originally analyze now require flow-sensitive analysis currently not
; supported in SCEV. In particular, during LV code generation, after turning
; the original scalar loop into the remainder loop, LV expected
; Legal->isConsecutivePtr() to be consistent and return the same output as
; during legal/cost model phases (original scalar loop). Unfortunately, that
; condition was not satisfied because of the aforementioned SCEV limitation.
; After D39346, LV code generation doesn't rely on Legal->isConsecutivePtr(),
; i.e., SCEV. This test verifies that LV is able to handle the described cases.
;
; TODO: The SCEV limitation described before may affect plans to further
; optimize the remainder loop of this particular test case. One tentative
; solution is to detect the problematic IVs in LV (%7 and %8) and perform an
; in-place IV optimization by replacing:
;   %8 = phi i32 [ %.ph2, %.outer ], [ %7, %6 ] with
; with
;   %8 = sub i32 %7, 1.


; Verify that store is vectorized as stride-1 memory access.

; CHECK-LABEL: @test_01(
; CHECK-NOT: vector.body:

; This test was originally vectorized, but now SCEV is smart enough to prove
; that its trip count is 1, so it gets ignored by vectorizer.
; Function Attrs: uwtable
define void @test_01() {
  br label %.outer

; <label>:1:                                      ; preds = %2
  ret void

; <label>:2:                                      ; preds = %._crit_edge.loopexit
  %3 = add nsw i32 %.ph, -2
  br i1 undef, label %1, label %.outer

.outer:                                           ; preds = %2, %0
  %.ph = phi i32 [ %3, %2 ], [ 336, %0 ]
  %.ph2 = phi i32 [ 62, %2 ], [ 110, %0 ]
  %4 = and i32 %.ph, 30
  %5 = add i32 %.ph2, 1
  br label %6

; <label>:6:                                      ; preds = %6, %.outer
  %7 = phi i32 [ %5, %.outer ], [ %13, %6 ]
  %8 = phi i32 [ %.ph2, %.outer ], [ %7, %6 ]
  %9 = add i32 %8, 2
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) undef, i64 %10
  %12 = ashr i32 undef, %4
  store i32 %12, ptr addrspace(1) %11, align 4
  %13 = add i32 %7, 1
  %14 = icmp sgt i32 %13, 61
  br i1 %14, label %._crit_edge.loopexit, label %6

._crit_edge.loopexit:                             ; preds = %._crit_edge.loopexit, %6
  br i1 undef, label %2, label %._crit_edge.loopexit
}

; After trip count is increased, the test gets vectorized.
; CHECK-LABEL: @test_02(
; CHECK: vector.body:
; CHECK: store <4 x i32>

; Function Attrs: uwtable
define void @test_02() {
  br label %.outer

; <label>:1:                                      ; preds = %2
  ret void

; <label>:2:                                      ; preds = %._crit_edge.loopexit
  %3 = add nsw i32 %.ph, -2
  br i1 undef, label %1, label %.outer

.outer:                                           ; preds = %2, %0
  %.ph = phi i32 [ %3, %2 ], [ 336, %0 ]
  %.ph2 = phi i32 [ 62, %2 ], [ 110, %0 ]
  %4 = and i32 %.ph, 30
  %5 = add i32 %.ph2, 1
  br label %6

; <label>:6:                                      ; preds = %6, %.outer
  %7 = phi i32 [ %5, %.outer ], [ %13, %6 ]
  %8 = phi i32 [ %.ph2, %.outer ], [ %7, %6 ]
  %9 = add i32 %8, 2
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) undef, i64 %10
  %12 = ashr i32 undef, %4
  store i32 %12, ptr addrspace(1) %11, align 4
  %13 = add i32 %7, 1
  %14 = icmp sgt i32 %13, 610
  br i1 %14, label %._crit_edge.loopexit, label %6

._crit_edge.loopexit:                             ; preds = %._crit_edge.loopexit, %6
  br i1 undef, label %2, label %._crit_edge.loopexit
}
