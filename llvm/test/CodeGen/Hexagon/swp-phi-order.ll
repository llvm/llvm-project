; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

%s.0 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, [49 x i8], [49 x i8], [25 x i8], [6 x i8], [29 x i8], i8, [6 x i8], [6 x i8] }

define void @f0(ptr nocapture %a0) {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v6, %b1 ], [ undef, %b0 ]
  %v1 = phi i32 [ %v8, %b1 ], [ 1, %b0 ]
  %v2 = and i32 %v0, 255
  %v3 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 9, i32 %v1
  %v4 = load i8, ptr %v3, align 1
  %v5 = zext i8 %v4 to i32
  %v6 = add nsw i32 %v5, %v2
  %v7 = trunc i32 %v6 to i8
  store i8 %v7, ptr %v3, align 1
  %v8 = add nsw i32 %v1, 1
  %v9 = icmp sgt i32 %v8, undef
  br i1 %v9, label %b2, label %b1

b2:                                               ; preds = %b2, %b1, %b0
  br label %b2
}
