; RUN: llc -march=hexagon -hexagon-hwloop-preheader < %s
; REQUIRES: asserts

; Checks that a subreg in a Phi is propagated correctly when a
; new preheader is created in the Hardware Loop pass.

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b2

b1:                                               ; preds = %b2, %b1
  %v0 = or i64 0, undef
  %v1 = add nsw i64 0, %v0
  %v2 = add nsw i64 %v1, 0
  %v3 = add nsw i64 %v2, 0
  %v4 = add nsw i64 %v3, 0
  %v5 = add nsw i64 %v4, 0
  %v6 = add nsw i64 %v5, 0
  %v7 = load i32, i32* undef, align 4
  %v8 = ashr i32 %v7, 5
  %v9 = sext i32 %v8 to i64
  %v10 = mul nsw i64 %v9, %v9
  %v11 = add nsw i64 %v6, %v10
  %v12 = add nsw i64 %v11, 0
  %v13 = add nsw i64 0, %v12
  %v14 = add nsw i64 %v13, 0
  %v15 = add nsw i64 %v14, 0
  %v16 = add nsw i64 %v15, 0
  %v17 = add nsw i64 %v16, 0
  %v18 = add nsw i64 %v17, 0
  %v19 = add nsw i64 %v18, 0
  %v20 = add nsw i64 %v19, 0
  %v21 = lshr i64 %v20, 32
  %v22 = trunc i64 %v21 to i32
  br i1 undef, label %b1, label %b3

b2:                                               ; preds = %b5, %b0
  br i1 undef, label %b1, label %b4

b3:                                               ; preds = %b1
  br i1 false, label %b5, label %b4

b4:                                               ; preds = %b4, %b3, %b2
  %v23 = phi i32 [ %v37, %b4 ], [ undef, %b2 ], [ %v22, %b3 ]
  %v24 = zext i32 %v23 to i64
  %v25 = shl nuw i64 %v24, 32
  %v26 = or i64 %v25, 0
  %v27 = add nsw i64 0, %v26
  %v28 = add nsw i64 %v27, 0
  %v29 = add nsw i64 %v28, 0
  %v30 = add nsw i64 %v29, 0
  %v31 = add nsw i64 %v30, 0
  %v32 = add nsw i64 %v31, 0
  %v33 = add nsw i64 %v32, 0
  %v34 = add nsw i64 %v33, 0
  %v35 = trunc i64 %v34 to i32
  %v36 = lshr i64 %v34, 32
  %v37 = trunc i64 %v36 to i32
  %v38 = icmp slt i32 undef, undef
  br i1 %v38, label %b4, label %b5

b5:                                               ; preds = %b4, %b3
  br label %b2
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
