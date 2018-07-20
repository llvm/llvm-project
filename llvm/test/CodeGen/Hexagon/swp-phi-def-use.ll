; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't assert when renaming a phi
; that looks like: a = PHI b, a

%s.0 = type { i32, i32*, [0 x i32], [0 x i32], [1 x i32] }
%s.1 = type { %s.2, %s.4, %s.5 }
%s.2 = type { %s.3 }
%s.3 = type { i32 }
%s.4 = type { i32 }
%s.5 = type { [0 x i32], [0 x i32 (i32*, i32*, i32*, i32*, i32*, i32, i32*)*] }

@g0 = external global i32, align 4
@g1 = external global %s.0, align 4
@g2 = external global i32, align 4
@g3 = external global i32, align 4
@g4 = external global i32*, align 4

define void @f0(%s.1* nocapture readonly %a0) #0 {
b0:
  %v0 = alloca [0 x i32], align 4
  %v1 = load i32, i32* @g0, align 4
  %v2 = load i32, i32* undef, align 4
  %v3 = load i32*, i32** getelementptr inbounds (%s.0, %s.0* @g1, i32 0, i32 1), align 4
  %v4 = load i32, i32* @g2, align 4
  %v5 = sub i32 0, %v4
  %v6 = getelementptr inbounds i32, i32* %v3, i32 %v5
  %v7 = load i32, i32* undef, align 4
  switch i32 %v7, label %b15 [
    i32 0, label %b1
    i32 1, label %b2
  ]

b1:                                               ; preds = %b0
  store i32 0, i32* @g3, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v8 = icmp eq i32 %v1, 0
  %v9 = icmp sgt i32 %v2, 0
  %v10 = getelementptr inbounds [0 x i32], [0 x i32]* %v0, i32 0, i32 0
  %v11 = sdiv i32 %v2, 2
  %v12 = add i32 %v11, -1
  %v13 = getelementptr inbounds [0 x i32], [0 x i32]* %v0, i32 0, i32 1
  %v14 = getelementptr inbounds %s.1, %s.1* %a0, i32 0, i32 2, i32 1, i32 %v1
  %v15 = sub i32 1, %v4
  %v16 = getelementptr inbounds i32, i32* %v3, i32 %v15
  %v17 = sdiv i32 %v2, 4
  %v18 = icmp slt i32 %v2, -3
  %v19 = add i32 %v2, -1
  %v20 = lshr i32 %v19, 2
  %v21 = mul i32 %v20, 4
  %v22 = add i32 %v21, 4
  %v23 = add i32 %v11, -2
  %v24 = add i32 %v17, 1
  %v25 = select i1 %v18, i32 1, i32 %v24
  br label %b4

b3:                                               ; preds = %b14
  store i32 %v25, i32* @g3, align 4
  br label %b4

b4:                                               ; preds = %b13, %b3, %b2
  %v26 = phi i32 [ undef, %b2 ], [ %v42, %b3 ], [ %v42, %b13 ]
  %v27 = phi i32 [ undef, %b2 ], [ 0, %b3 ], [ 0, %b13 ]
  %v28 = phi i32 [ undef, %b2 ], [ %v30, %b3 ], [ %v30, %b13 ]
  %v29 = phi i32 [ undef, %b2 ], [ %v43, %b3 ], [ %v43, %b13 ]
  %v30 = phi i32 [ undef, %b2 ], [ undef, %b3 ], [ 0, %b13 ]
  br i1 %v8, label %b6, label %b5

b5:                                               ; preds = %b5, %b4
  br label %b5

b6:                                               ; preds = %b4
  br i1 %v9, label %b8, label %b7

b7:                                               ; preds = %b6
  store i32 0, i32* @g3, align 4
  br label %b11

b8:                                               ; preds = %b6
  br i1 undef, label %b9, label %b11

b9:                                               ; preds = %b8
  %v31 = load i32*, i32** @g4, align 4
  br label %b10

b10:                                              ; preds = %b10, %b9
  %v32 = phi i32 [ %v22, %b9 ], [ %v39, %b10 ]
  %v33 = phi i32 [ %v29, %b9 ], [ %v38, %b10 ]
  %v34 = add nsw i32 %v32, %v28
  %v35 = shl i32 %v34, 1
  %v36 = getelementptr inbounds i32, i32* %v31, i32 %v35
  %v37 = load i32, i32* %v36, align 4
  %v38 = select i1 false, i32 0, i32 %v33
  %v39 = add nsw i32 %v32, 1
  store i32 %v39, i32* @g3, align 4
  %v40 = icmp slt i32 %v39, 0
  br i1 %v40, label %b10, label %b11

b11:                                              ; preds = %b10, %b8, %b7
  %v41 = phi i32 [ %v29, %b8 ], [ %v29, %b7 ], [ %v38, %b10 ]
  br i1 false, label %b12, label %b13

b12:                                              ; preds = %b11
  br label %b13

b13:                                              ; preds = %b12, %b11
  %v42 = load i32, i32* %v10, align 4
  %v43 = select i1 false, i32 %v41, i32 1
  br i1 %v18, label %b4, label %b14

b14:                                              ; preds = %b14, %b13
  br i1 false, label %b14, label %b3

b15:                                              ; preds = %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
