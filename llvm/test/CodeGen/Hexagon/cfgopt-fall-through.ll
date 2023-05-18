; RUN: llc -march=hexagon -verify-machineinstrs -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; REQUIRES: asserts

; Check for some sane output. This test used to crash.
; CHECK: jumpr r31


define i32 @fred(i32 %a0, i8 zeroext %a1) local_unnamed_addr #0 {
b2:
  br i1 undef, label %b4, label %b3

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; preds = %b2
  br i1 undef, label %b19, label %b5

b5:                                               ; preds = %b4
  br i1 undef, label %b6, label %b12

b6:                                               ; preds = %b5
  switch i8 %a1, label %b17 [
    i8 2, label %b7
    i8 5, label %b7
    i8 1, label %b7
    i8 3, label %b8
  ]

b7:                                               ; preds = %b6, %b6, %b6
  unreachable

b8:                                               ; preds = %b6
  br i1 undef, label %b11, label %b9

b9:                                               ; preds = %b8
  %v10 = or i32 undef, 0
  br label %b15

b11:                                              ; preds = %b8
  unreachable

b12:                                              ; preds = %b5
  switch i8 %a1, label %b17 [
    i8 5, label %b13
    i8 1, label %b13
    i8 2, label %b14
    i8 3, label %b15
  ]

b13:                                              ; preds = %b12, %b12
  store i32 %a0, ptr undef, align 4
  br label %b17

b14:                                              ; preds = %b12
  store i16 undef, ptr undef, align 4
  br label %b17

b15:                                              ; preds = %b12, %b9
  %v16 = phi i32 [ 0, %b12 ], [ %v10, %b9 ]
  store i32 undef, ptr undef, align 4
  br label %b17

b17:                                              ; preds = %b15, %b14, %b13, %b12, %b6
  %v18 = phi i32 [ 0, %b13 ], [ 0, %b12 ], [ %v16, %b15 ], [ 0, %b14 ], [ 0, %b6 ]
  ret i32 %v18

b19:                                              ; preds = %b4
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" "target-features"="-hvx,-long-calls" }
