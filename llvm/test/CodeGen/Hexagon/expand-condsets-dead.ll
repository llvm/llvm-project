; RUN: llc -mtriple=hexagon -O2 < %s
; REQUIRES: asserts

; Test that the dead and kill flags are not added incorrectly during the
; Hexagon Expand Condsets pass. The pass shouldn't add a kill flag to a use that
; is tied to a definition, and the pass shouldn't remove the dead flag for a
; definition that is really dead. The removal of the dead flag causes an assert
; in the Machine Scheduler when querying live interval information.

define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v0 = load i16, ptr undef, align 4
  %v1 = sext i16 %v0 to i32
  %v2 = and i32 %v1, 7
  %v3 = sub nsw i32 8, %v2
  %v4 = sub nsw i32 8, 0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v5 = phi ptr [ undef, %b1 ], [ %v16, %b2 ]
  %v6 = phi i32 [ 4, %b1 ], [ %v17, %b2 ]
  %v7 = load i8, ptr undef, align 1
  %v8 = zext i8 %v7 to i32
  %v9 = mul nuw nsw i32 %v8, %v3
  %v10 = add nuw nsw i32 0, %v9
  %v11 = mul nuw nsw i32 %v10, %v4
  %v12 = add nuw nsw i32 0, %v11
  %v13 = lshr i32 %v12, 6
  %v14 = trunc i32 %v13 to i8
  store i8 %v14, ptr %v5, align 1
  %v15 = getelementptr inbounds i8, ptr %v5, i32 1
  %v16 = select i1 undef, ptr undef, ptr %v15
  %v17 = add nsw i32 %v6, -1
  %v18 = icmp eq i32 %v17, 0
  br i1 %v18, label %b3, label %b2

b3:                                               ; preds = %b2
  br i1 undef, label %b1, label %b4

b4:                                               ; preds = %b3
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" }
