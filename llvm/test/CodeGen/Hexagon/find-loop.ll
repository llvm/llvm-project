; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; Test that the compiler doesn't assert when attempting to find a
; loop instruction that has been deleted, so FindLoopInstr returns
; the loop instruction from a different loop.

@g0 = external global i32

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = alloca i64, align 8
  %v2 = load i32, ptr @g0, align 4
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v3 = phi i32 [ %v4, %b1 ], [ 64, %b0 ]
  %v4 = add nsw i32 %v3, 1
  %v5 = icmp slt i32 %v4, %v2
  br i1 %v5, label %b1, label %b2

b2:                                               ; preds = %b6, %b3, %b1, %b0
  br label %b4

b3:                                               ; preds = %b4
  br i1 undef, label %b4, label %b2

b4:                                               ; preds = %b3, %b2
  %v6 = icmp slt i32 undef, 1
  br i1 %v6, label %b3, label %b5

b5:                                               ; preds = %b5, %b4
  %v7 = phi i32 [ %v18, %b5 ], [ 1, %b4 ]
  %v8 = phi i32 [ %v19, %b5 ], [ 0, %b4 ]
  %v9 = add nsw i32 %v8, 0
  %v10 = lshr i32 %v9, 5
  %v11 = getelementptr inbounds [2 x i32], ptr %v0, i32 0, i32 %v10
  %v12 = load i32, ptr %v11, align 4
  %v13 = and i32 %v9, 31
  %v14 = shl i32 1, %v13
  %v15 = and i32 %v12, %v14
  %v16 = icmp ne i32 %v15, 0
  %v17 = zext i1 %v16 to i32
  %v18 = and i32 %v17, %v7
  %v19 = add nsw i32 %v8, 1
  %v20 = icmp eq i32 %v19, 1
  br i1 %v20, label %b6, label %b5

b6:                                               ; preds = %b5
  %v21 = icmp eq i32 %v18, 0
  br i1 %v21, label %b2, label %b7

b7:                                               ; preds = %b6
  tail call void @f1() #1
  unreachable
}

; Function Attrs: nounwind
declare void @f1() #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind }
