; RUN: llc -mtriple=hexagon -enable-aa-sched-mi -enable-pipeliner \
; RUN:     -hexagon-expand-condsets=0 -pipeliner-max-stages=2 < %s
; REQUIRES: asserts

; Disable expand-condsets because it will assert on undefined registers.

; Test that we generate pipelines with multiple stages correctly.

%s.0 = type { [194 x i32], ptr, [10 x i32], [10 x i32], i32, i32, i32, i32, i32, [9 x i32], [9 x i32], i16, i16, i16, i16, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%s.1 = type { [60 x i32], i16 }
%s.2 = type { i32, [7 x i32], i16 }
%s.3 = type { [10 x i32] }
%s.4 = type { [10 x i32], [10 x i32] }
%s.5 = type { [5 x i32], i32, i32 }
%s.6 = type { [5 x i32], i32, i32 }
%s.7 = type { [4 x i32], [4 x i32] }
%s.8 = type { [5 x i32], i32, i32, i16, i16 }
%s.9 = type { i8, i32, i32, i32, [10 x i32], [10 x i32], [80 x i32], [80 x i32], [8 x i32], i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }

; Function Attrs: nounwind
define fastcc void @f0(ptr %a0) #0 {
b0:
  %v0 = alloca [40 x i32], align 8
  %v1 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 5
  %v2 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 6
  %v3 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 4
  %v4 = select i1 undef, ptr %v2, ptr %v1
  %v5 = load i32, ptr %v4, align 4
  br i1 false, label %b2, label %b1

b1:                                               ; preds = %b0
  %v6 = load i32, ptr %v3, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v7 = phi i32 [ %v6, %b1 ], [ undef, %b0 ]
  %v8 = shl i32 %v7, 1
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b3, %b2
  %v9 = phi i32 [ %v34, %b3 ], [ %v5, %b2 ]
  %v10 = add nsw i32 %v9, 2
  %v11 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 undef
  %v12 = load i32, ptr %v11, align 4
  %v13 = mul nsw i32 %v12, %v8
  %v14 = ashr i32 %v13, 15
  %v15 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 %v10
  %v16 = add nsw i32 %v14, 0
  store i32 %v16, ptr %v15, align 4
  %v17 = add nsw i32 %v9, 3
  %v18 = sub nsw i32 %v17, %v5
  %v19 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 %v18
  %v20 = load i32, ptr %v19, align 4
  %v21 = mul nsw i32 %v20, %v8
  %v22 = ashr i32 %v21, 15
  %v23 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 %v17
  %v24 = add nsw i32 %v22, 0
  store i32 %v24, ptr %v23, align 4
  %v25 = add nsw i32 %v9, 6
  %v26 = sub nsw i32 %v25, %v5
  %v27 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 %v26
  %v28 = load i32, ptr %v27, align 4
  %v29 = mul nsw i32 %v28, %v8
  %v30 = ashr i32 %v29, 15
  %v31 = getelementptr inbounds [40 x i32], ptr %v0, i32 0, i32 %v25
  %v32 = load i32, ptr %v31, align 4
  %v33 = add nsw i32 %v30, %v32
  store i32 %v33, ptr %v31, align 4
  %v34 = add nsw i32 %v9, 8
  %v35 = icmp slt i32 %v34, 33
  br i1 %v35, label %b3, label %b4

b4:                                               ; preds = %b4, %b3, %b2
  br label %b4
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
