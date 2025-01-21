; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't assert in orderDependence because
; the check for OrderAfterDef precedeence is in the wrong spot.

%s.0 = type <{ i8, [20 x %s.1] }>
%s.1 = type { i16, i16 }

; Function Attrs: nounwind optsize ssp
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v3, %b1 ], [ 0, %b0 ]
  %v1 = getelementptr inbounds %s.0, ptr undef, i32 0, i32 1, i32 %v0, i32 0
  store i16 0, ptr %v1, align 1
  %v2 = getelementptr inbounds %s.0, ptr undef, i32 0, i32 1, i32 %v0, i32 1
  store i16 -1, ptr %v2, align 1
  %v3 = add nsw i32 %v0, 1
  %v4 = icmp eq i32 %v3, 20
  br i1 %v4, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind optsize ssp "target-cpu"="hexagonv55" }
