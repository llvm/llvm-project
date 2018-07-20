; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate multiple using upper result.

; CHECK: = mpy(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: = mpy(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: = mpy(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: = mpy(r{{[0-9]+}},r{{[0-9]+}})

@g0 = external constant [1152 x i32], align 8
@g1 = external constant [2 x i32], align 8

; Function Attrs: nounwind
define void @f0(i32* nocapture readonly %a0, i32* %a1, i32* nocapture %a2, i32 %a3, i32 %a4) #0 {
b0:
  %v0 = getelementptr inbounds i32, i32* %a0, i32 512
  %v1 = getelementptr inbounds i32, i32* %a0, i32 511
  %v2 = getelementptr inbounds i32, i32* %a2, i32 1023
  %v3 = getelementptr inbounds i32, i32* %a1, i32 1023
  br label %b1

b1:                                               ; preds = %b0
  %v4 = load i32, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @g1, i32 0, i32 1), align 4
  %v5 = getelementptr inbounds [1152 x i32], [1152 x i32]* @g0, i32 0, i32 %v4
  br label %b2

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v6 = phi i32* [ %v30, %b3 ], [ %a2, %b2 ]
  %v7 = phi i32* [ %v44, %b3 ], [ %a1, %b2 ]
  %v8 = phi i32* [ %v17, %b3 ], [ %v0, %b2 ]
  %v9 = phi i32* [ %v34, %b3 ], [ %v1, %b2 ]
  %v10 = phi i32* [ %v40, %b3 ], [ %v3, %b2 ]
  %v11 = phi i32* [ %v33, %b3 ], [ %v2, %b2 ]
  %v12 = phi i32* [ %v15, %b3 ], [ %v5, %b2 ]
  %v13 = getelementptr inbounds i32, i32* %v12, i32 1
  %v14 = load i32, i32* %v12, align 4
  %v15 = getelementptr inbounds i32, i32* %v12, i32 2
  %v16 = load i32, i32* %v13, align 4
  %v17 = getelementptr inbounds i32, i32* %v8, i32 1
  %v18 = load i32, i32* %v8, align 4
  %v19 = sext i32 %v14 to i64
  %v20 = sext i32 %v18 to i64
  %v21 = mul nsw i64 %v20, %v19
  %v22 = lshr i64 %v21, 32
  %v23 = trunc i64 %v22 to i32
  %v24 = sext i32 %v16 to i64
  %v25 = mul nsw i64 %v20, %v24
  %v26 = lshr i64 %v25, 32
  %v27 = trunc i64 %v26 to i32
  %v28 = load i32, i32* %v7, align 4
  %v29 = sub nsw i32 %v28, %v23
  %v30 = getelementptr inbounds i32, i32* %v6, i32 1
  store i32 %v29, i32* %v6, align 4
  %v31 = load i32, i32* %v10, align 4
  %v32 = add nsw i32 %v27, %v31
  %v33 = getelementptr inbounds i32, i32* %v11, i32 -1
  store i32 %v32, i32* %v11, align 4
  %v34 = getelementptr inbounds i32, i32* %v9, i32 -1
  %v35 = load i32, i32* %v9, align 4
  %v36 = sext i32 %v35 to i64
  %v37 = mul nsw i64 %v36, %v19
  %v38 = lshr i64 %v37, 32
  %v39 = trunc i64 %v38 to i32
  %v40 = getelementptr inbounds i32, i32* %v10, i32 -1
  store i32 %v39, i32* %v10, align 4
  %v41 = mul nsw i64 %v36, %v24
  %v42 = lshr i64 %v41, 32
  %v43 = trunc i64 %v42 to i32
  %v44 = getelementptr inbounds i32, i32* %v7, i32 1
  store i32 %v43, i32* %v7, align 4
  %v45 = icmp ult i32* %v44, %v40
  br i1 %v45, label %b3, label %b4

b4:                                               ; preds = %b3
  br label %b6

b5:                                               ; No predecessors!
  br label %b6

b6:                                               ; preds = %b5, %b4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
