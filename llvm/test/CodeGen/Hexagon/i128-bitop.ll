; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { i32, i32, [10 x %s.1] }
%s.1 = type { [4 x i32] }
%s.2 = type { i128 }

@g0 = external global %s.0*

; Function Attrs: nounwind ssp
define void @f0(%s.2* nocapture %a0, i32 %a1) #0 {
b0:
  %v0 = getelementptr inbounds %s.2, %s.2* %a0, i32 0, i32 0
  br label %b1

b1:                                               ; preds = %b4, %b3, %b0
  %v1 = phi i32 [ 0, %b0 ], [ %v14, %b4 ], [ %v13, %b3 ]
  switch i32 %v1, label %b4 [
    i32 0, label %b3
    i32 1, label %b2
  ]

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v2 = phi i32 [ 1, %b2 ], [ 0, %b1 ]
  %v3 = phi i128 [ 64, %b2 ], [ 32, %b1 ]
  %v4 = phi i128 [ -79228162495817593519834398721, %b2 ], [ -18446744069414584321, %b1 ]
  %v5 = load %s.0*, %s.0** @g0, align 4
  %v6 = getelementptr inbounds %s.0, %s.0* %v5, i32 0, i32 2, i32 %a1, i32 0, i32 %v2
  %v7 = load i32, i32* %v6, align 4
  %v8 = zext i32 %v7 to i128
  %v9 = load i128, i128* %v0, align 4
  %v10 = shl nuw nsw i128 %v8, %v3
  %v11 = and i128 %v9, %v4
  %v12 = or i128 %v11, %v10
  store i128 %v12, i128* %v0, align 4
  %v13 = add i32 %v1, 1
  br label %b1

b4:                                               ; preds = %b1
  %v14 = add i32 %v1, 1
  %v15 = icmp eq i32 %v14, 4
  br i1 %v15, label %b5, label %b1

b5:                                               ; preds = %b4
  ret void
}

; Function Attrs: nounwind ssp
define void @f1(%s.2* nocapture %a0, i32 %a1) #0 {
b0:
  %v0 = getelementptr inbounds %s.2, %s.2* %a0, i32 0, i32 0
  br label %b1

b1:                                               ; preds = %b5, %b4, %b0
  %v1 = phi i32 [ 0, %b0 ], [ %v20, %b5 ], [ %v19, %b4 ]
  switch i32 %v1, label %b5 [
    i32 0, label %b2
    i32 1, label %b3
  ]

b2:                                               ; preds = %b1
  %v2 = load %s.0*, %s.0** @g0, align 4
  %v3 = getelementptr inbounds %s.0, %s.0* %v2, i32 0, i32 2, i32 %a1, i32 0, i32 0
  %v4 = load i32, i32* %v3, align 4
  %v5 = zext i32 %v4 to i128
  %v6 = load i128, i128* %v0, align 4
  %v7 = shl nuw nsw i128 %v5, 32
  %v8 = and i128 %v6, -18446744069414584321
  %v9 = or i128 %v8, %v7
  br label %b4

b3:                                               ; preds = %b1
  %v10 = load %s.0*, %s.0** @g0, align 4
  %v11 = getelementptr inbounds %s.0, %s.0* %v10, i32 0, i32 2, i32 %a1, i32 0, i32 1
  %v12 = load i32, i32* %v11, align 4
  %v13 = zext i32 %v12 to i128
  %v14 = load i128, i128* %v0, align 4
  %v15 = shl nuw nsw i128 %v13, 64
  %v16 = and i128 %v14, -79228162495817593519834398721
  %v17 = or i128 %v16, %v15
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v18 = phi i128 [ %v17, %b3 ], [ %v9, %b2 ]
  store i128 %v18, i128* %v0, align 4
  %v19 = add i32 %v1, 1
  br label %b1

b5:                                               ; preds = %b1
  %v20 = add i32 %v1, 1
  %v21 = icmp eq i32 %v20, 4
  br i1 %v21, label %b6, label %b1

b6:                                               ; preds = %b5
  ret void
}

attributes #0 = { nounwind ssp }
