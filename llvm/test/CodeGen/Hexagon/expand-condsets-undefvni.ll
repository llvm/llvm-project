; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Check that this compiles successfully.
; CHECK: jumpr r31

target triple = "hexagon"

define i64 @fred(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b2:
  %v3 = lshr i64 %a1, 52
  %v4 = trunc i64 %v3 to i11
  switch i11 %v4, label %b15 [
    i11 -1, label %b5
    i11 0, label %b14
  ]

b5:                                               ; preds = %b2
  br i1 undef, label %b13, label %b6

b6:                                               ; preds = %b5
  %v7 = or i64 %a1, 2251799813685248
  br i1 undef, label %b8, label %b10

b8:                                               ; preds = %b6
  %v9 = select i1 undef, i64 %v7, i64 undef
  br label %b16

b10:                                              ; preds = %b6
  br i1 undef, label %b16, label %b11

b11:                                              ; preds = %b10
  %v12 = select i1 undef, i64 undef, i64 %v7
  br label %b16

b13:                                              ; preds = %b5
  br label %b16

b14:                                              ; preds = %b2
  br label %b16

b15:                                              ; preds = %b2
  br label %b16

b16:                                              ; preds = %b15, %b14, %b13, %b11, %b10, %b8
  %v17 = phi i64 [ undef, %b13 ], [ -2251799813685248, %b14 ], [ 0, %b15 ], [ %v12, %b11 ], [ %v9, %b8 ], [ %v7, %b10 ]
  ret i64 %v17
}

attributes #0 = { nounwind "target-cpu"="hexagonv62" }
