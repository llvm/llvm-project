; RUN: llc -march=hexagon -enable-pipeliner -stats -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; Check that we handle the case when a value is first defined in the loop.

; STATS: 1 pipeliner        - Number of loops software pipelined

; Function Attrs: nounwind
define fastcc void @f0() #0 {
b0:
  br i1 undef, label %b7, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b4

b2:                                               ; preds = %b1
  %v0 = load i16, i16* undef, align 2
  %v1 = load i16, i16* undef, align 2
  br i1 undef, label %b5, label %b3

b3:                                               ; preds = %b5, %b2
  %v2 = phi i16 [ 0, %b2 ], [ %v14, %b5 ]
  br label %b4

b4:                                               ; preds = %b3, %b1
  br i1 undef, label %b7, label %b6

b5:                                               ; preds = %b5, %b2
  %v3 = phi i16 [ %v5, %b5 ], [ undef, %b2 ]
  %v4 = phi i16 [ 0, %b5 ], [ %v1, %b2 ]
  %v5 = phi i16 [ 0, %b5 ], [ %v0, %b2 ]
  %v6 = phi i16 [ %v4, %b5 ], [ undef, %b2 ]
  %v7 = phi i16 [ %v14, %b5 ], [ 0, %b2 ]
  %v8 = phi i32 [ %v15, %b5 ], [ undef, %b2 ]
  %v9 = or i16 0, %v7
  %v10 = lshr i16 %v3, 8
  %v11 = lshr i16 %v6, 8
  %v12 = or i16 %v11, %v9
  %v13 = or i16 0, %v12
  %v14 = or i16 %v10, %v13
  %v15 = add nsw i32 %v8, -32
  %v16 = icmp sgt i32 %v15, 31
  br i1 %v16, label %b5, label %b3

b6:                                               ; preds = %b4
  br label %b7

b7:                                               ; preds = %b6, %b4, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
