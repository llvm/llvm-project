; RUN: opt -S -unroll-runtime -passes=loop-unroll < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;; Check that we don't emit expensive instructions to compute trip
;; counts when unrolling loops.

define i32 @test(i64 %v12, i8* %array, i64* %loc) {
; CHECK-LABEL: @test(
; CHECK-NOT: udiv
entry:
  %step = load i64, i64* %loc, !range !0
  br label %loop

loop:                                           ; preds = %entry, %loop
  %k.015 = phi i64 [ %v15, %loop ], [ %v12, %entry ]
  %v14 = getelementptr inbounds i8, i8* %array, i64 %k.015
  store i8 0, i8* %v14
  %v15 = add nuw nsw i64 %k.015, %step
  %v16 = icmp slt i64 %v15, 8193
  br i1 %v16, label %loop, label %loopexit

loopexit:                             ; preds = %loop
  ret i32 0
}

;; Though SCEV for loop tripcount contains division,
;; it shouldn't be considered expensive, since the division already
;; exists in the code and we don't need to expand it once more.
;; Thus, it shouldn't prevent us from unrolling the loop.

define i32 @test2(i64* %loc, i64 %conv7) {
; CHECK-LABEL: @test2(
; CHECK: udiv
; CHECK: udiv
; CHECK-NOT: udiv
; CHECK-LABEL: for.body
entry:
  %rem0 = load i64, i64* %loc, align 8
  %ExpensiveComputation = udiv i64 %rem0, 42 ; <<< Extra computations are added to the trip-count expression
  br label %bb1
bb1:
  %div11 = udiv i64 %ExpensiveComputation, %conv7
  %cmp.i38 = icmp ugt i64 %div11, 1
  %div12 = select i1 %cmp.i38, i64 %div11, i64 1
  br label %for.body
for.body:
  %rem1 = phi i64 [ %rem0, %bb1 ], [ %rem2, %for.body ]
  %k1 = phi i64 [ %div12, %bb1 ], [ %dec, %for.body ]
  %mul1 = mul i64 %rem1, 48271
  %rem2 = urem i64 %mul1, 2147483647
  %dec = add i64 %k1, -1
  %cmp = icmp eq i64 %dec, 0
  br i1 %cmp, label %exit, label %for.body
exit:
  %rem3 = phi i64 [ %rem2, %for.body ]
  store i64 %rem3, i64* %loc, align 8
  ret i32 0
}

!0 = !{i64 1, i64 100}
