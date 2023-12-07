; REQUIRES: asserts
; RUN: opt -loop-reduce -debug-only=loop-reduce -S  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"
;
; %lsr.iv2 and %lsr.iv10 are in same bb, but they are not equal since start
; value are different.
;
; %scevgep = getelementptr [0 x %0], [0 x %0]* %arg, i64 0, i64 99
; %scevgep1 = bitcast %0* %scevgep to [0 x %0]*
; %lsr.iv2 = phi [0 x %0]* [ %1, %bb18 ], [ %scevgep1, %bb ]
;
; %lsr.iv10 = phi [0 x %0]* [ %2, %bb18 ], [ %arg, %bb ]
;
; Make sure two incomplete phis will not be marked as congruent.
;
; CHECK: One incomplete PHI is found:   %[[IV:.*]] = phi ptr
; CHECK: One incomplete PHI is found:   %[[IV2:.*]] = phi ptr
; CHECK-NOT: Eliminated congruent iv:  %[[IV]]
; CHECK-NOT: Original iv: %[[IV2]]
; CHECK-NOT: Eliminated congruent iv:  %[[IV2]]
; CHECK-NOT: Original iv: %[[IV]]

%0 = type <{ float }>

define void @foo(ptr %arg) {
bb:
  %i = getelementptr inbounds [0 x %0], ptr %arg, i64 0, i64 -1
  %i2 = getelementptr i8, ptr %i, i64 4
  br label %bb3

bb3:                                              ; preds = %bb18, %bb
  %i4 = phi i64 [ %i20, %bb18 ], [ 0, %bb ]
  %i5 = phi i64 [ %i21, %bb18 ], [ 1, %bb ]
  br i1 undef, label %bb22, label %bb9

bb9:                                              ; preds = %bb9, %bb3
  %i10 = phi i64 [ 0, %bb3 ], [ %i16, %bb9 ]
  %i11 = add i64 %i10, %i4
  %i12 = shl i64 %i11, 2
  %i13 = getelementptr i8, ptr %i2, i64 %i12
  store <4 x float> undef, ptr %i13, align 4
  %i16 = add i64 %i10, 32
  br i1 true, label %bb17, label %bb9

bb17:                                             ; preds = %bb9
  br i1 undef, label %bb18, label %bb22

bb18:                                             ; preds = %bb17
  %i19 = add i64 undef, %i4
  %i20 = add i64 %i19, %i5
  %i21 = add nuw nsw i64 %i5, 1
  br label %bb3

bb22:                                             ; preds = %bb22, %bb17, %bb3
  %i23 = phi i64 [ %i26, %bb22 ], [ undef, %bb17 ], [ 100, %bb3 ]
  %i24 = add nsw i64 %i23, %i4
  %i25 = getelementptr %0, ptr %i, i64 %i24, i32 0
  store float undef, ptr %i25, align 4
  %i26 = add nuw nsw i64 %i23, 1
  br label %bb22
}
