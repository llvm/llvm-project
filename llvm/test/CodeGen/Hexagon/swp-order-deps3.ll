; RUN: llc -mtriple=hexagon -O2 -simplifycfg-require-and-preserve-domtree=1 < %s
; REQUIRES: asserts

; Function Attrs: noinline nounwind ssp
define fastcc void @f0() #0 {
b0:
  %v0 = add i32 0, 39
  %v1 = and i32 %v0, -8
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ %v10, %b1 ], [ undef, %b0 ]
  %v3 = phi ptr [ %v7, %b1 ], [ undef, %b0 ]
  %v4 = ptrtoint ptr %v3 to i32
  %v5 = add i32 %v4, %v1
  store i32 %v5, ptr %v3, align 4
  %v7 = getelementptr inbounds i8, ptr %v3, i32 %v1
  store i32 1111638594, ptr %v3, align 4
  %v10 = add nsw i32 %v2, -1
  %v11 = icmp sgt i32 %v10, 0
  br i1 %v11, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { noinline nounwind ssp "target-cpu"="hexagonv55" }
