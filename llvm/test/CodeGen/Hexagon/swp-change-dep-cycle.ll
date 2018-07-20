; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; Don't change the dependences if it's going to cause a cycle.

; Function Attrs: nounwind
define void @f0(i8* nocapture %a0, i32 %a1) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i8* [ undef, %b1 ], [ undef, %b0 ]
  %v1 = phi i32 [ %v20, %b1 ], [ 1, %b0 ]
  %v2 = phi i8* [ %v6, %b1 ], [ %a0, %b0 ]
  %v3 = load i8, i8* %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = mul nsw i32 %v4, 3
  %v6 = getelementptr inbounds i8, i8* %v2, i32 1
  %v7 = load i8, i8* %v6, align 1
  %v8 = zext i8 %v7 to i32
  %v9 = add i32 %v8, 2
  %v10 = add i32 %v9, %v5
  %v11 = lshr i32 %v10, 2
  %v12 = trunc i32 %v11 to i8
  %v13 = getelementptr inbounds i8, i8* undef, i32 2
  store i8 %v12, i8* %v0, align 1
  %v14 = load i8, i8* %v2, align 1
  %v15 = zext i8 %v14 to i32
  %v16 = add i32 %v15, 2
  %v17 = add i32 %v16, 0
  %v18 = lshr i32 %v17, 2
  %v19 = trunc i32 %v18 to i8
  store i8 %v19, i8* %v13, align 1
  %v20 = add i32 %v1, 1
  %v21 = icmp eq i32 %v20, %a1
  br i1 %v21, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
