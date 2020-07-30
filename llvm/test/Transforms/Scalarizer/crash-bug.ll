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
;CHECK: %bb2_vec.i1 = phi i16 [ 200, %0 ], [ %bb2_vec.i1, %bb2 ]
  br i1 undef, label %bb3, label %bb2

bb3:
  ret void
}

; See https://reviews.llvm.org/D83101#2135945
define void @f1_crash(<2 x i16> %base, i1 %c, <2 x i16>* %ptr) {
; CHECK-LABEL: @f1_crash(
; CHECK: vector.ph:
; CHECK:   %base.i0 = extractelement <2 x i16> %base, i32 0
; CHECK:   %base.i1 = extractelement <2 x i16> %base, i32 1
; CHECK:   br label %vector.body115
; CHECK: vector.body115:                                   ; preds = %vector.body115, %vector.ph
; CHECK:   %vector.recur.i0 = phi i16 [ %base.i0, %vector.ph ], [ %wide.load125.i0, %vector.body115 ]
; CHECK:   %vector.recur.i1 = phi i16 [ %base.i1, %vector.ph ], [ %wide.load125.i1, %vector.body115 ]
; CHECK:   %wide.load125 = load <2 x i16>, <2 x i16>* %ptr, align 1
; CHECK:   %wide.load125.i0 = extractelement <2 x i16> %wide.load125, i32 0
; CHECK:   %wide.load125.i1 = extractelement <2 x i16> %wide.load125, i32 1
; CHECK:   br i1 %c, label %middle.block113, label %vector.body115
; CHECK: middle.block113:                                  ; preds = %vector.body115
; CHECK:   ret void
; CHECK: }

vector.ph:
  br label %vector.body115

vector.body115:
  %vector.recur = phi <2 x i16> [ %base, %vector.ph ], [ %wide.load125, %vector.body115 ]
  %wide.load125 = load <2 x i16>, <2 x i16>* %ptr, align 1
  br i1 %c, label %middle.block113, label %vector.body115

middle.block113:
  ret void
}
