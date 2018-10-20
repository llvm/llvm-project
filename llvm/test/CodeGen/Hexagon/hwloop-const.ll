; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: endloop

target triple = "hexagon-unknown-linux-gnu"

@g0 = common global [25000 x i32] zeroinitializer, align 8
@g1 = common global [25000 x i32] zeroinitializer, align 8

define i32 @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v3, %b1 ]
  %v1 = getelementptr inbounds [25000 x i32], [25000 x i32]* @g0, i32 0, i32 %v0
  store i32 %v0, i32* %v1, align 4
  %v2 = getelementptr inbounds [25000 x i32], [25000 x i32]* @g1, i32 0, i32 %v0
  store i32 %v0, i32* %v2, align 4
  %v3 = add nsw i32 %v0, 1
  %v4 = icmp eq i32 %v3, 25000
  br i1 %v4, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
