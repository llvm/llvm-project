; RUN: opt %s -scalarizer -S -o - | FileCheck %s
; RUN: opt %s -passes='function(scalarizer)' -S -o - | FileCheck %s

; Don't crash

define void @foo() {
  br label %bb1

bb2:                                        ; preds = %bb1
  %bb2_vec = shufflevector <2 x i16> <i16 0, i16 10000>,
                           <2 x i16> %bb1_vec,
                           <2 x i32> <i32 0, i32 3>
  br label %bb1

bb1:                                        ; preds = %bb2, %0
  %bb1_vec = phi <2 x i16> [ <i16 100, i16 200>, %0 ], [ %bb2_vec, %bb2 ]
;CHECK: bb1:
;CHECK: %bb1_vec.i0 = phi i16 [ 100, %0 ], [ 0, %bb2 ]
;CHECK: %bb2_vec.i1 = phi i16 [ 200, %0 ], [ %bb2_vec.i1, %bb2 ]
  br i1 undef, label %bb3, label %bb2

bb3:
  ret void
}

