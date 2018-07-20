; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure that we can handle loops with multiple ENDLOOP instructions.
; This situation can arise due to tail duplication.

; CHECK: loop1([[LP:.LBB0_[0-9]+]]
; CHECK: [[LP]]:
; CHECK-NOT: loop1(
; CHECK: endloop1
; CHECK: endloop1

%s.0 = type { i32, i8* }
%s.1 = type { i32, i32, i32, i32 }

define void @f0(%s.0* nocapture readonly %a0, %s.1* nocapture readonly %a1) {
b0:
  %v0 = getelementptr inbounds %s.1, %s.1* %a1, i32 0, i32 0
  %v1 = load i32, i32* %v0, align 4
  %v2 = getelementptr inbounds %s.1, %s.1* %a1, i32 0, i32 3
  %v3 = load i32, i32* %v2, align 4
  %v4 = getelementptr inbounds %s.1, %s.1* %a1, i32 0, i32 2
  %v5 = load i32, i32* %v4, align 4
  %v6 = getelementptr inbounds %s.1, %s.1* %a1, i32 0, i32 1
  %v7 = load i32, i32* %v6, align 4
  %v8 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 1
  %v9 = load i8*, i8** %v8, align 4
  %v10 = bitcast i8* %v9 to i32*
  %v11 = mul i32 %v1, 10
  %v12 = icmp eq i32 %v1, %v3
  %v13 = icmp eq i32 %v5, 0
  br i1 %v12, label %b3, label %b1

b1:                                               ; preds = %b0
  br i1 %v13, label %b14, label %b2

b2:                                               ; preds = %b1
  %v14 = lshr i32 %v11, 5
  %v15 = getelementptr inbounds i32, i32* %v10, i32 %v14
  %v16 = and i32 %v11, 30
  %v17 = icmp eq i32 %v16, 0
  br label %b11

b3:                                               ; preds = %b0
  br i1 %v13, label %b14, label %b4

b4:                                               ; preds = %b3
  %v18 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 0
  br label %b5

b5:                                               ; preds = %b6, %b4
  %v19 = phi i32 [ %v11, %b4 ], [ %v22, %b6 ]
  %v20 = phi i32 [ %v5, %b4 ], [ %v21, %b6 ]
  %v21 = add i32 %v20, -1
  %v22 = add i32 %v19, -10
  %v23 = lshr i32 %v22, 5
  %v24 = getelementptr inbounds i32, i32* %v10, i32 %v23
  %v25 = and i32 %v22, 31
  %v26 = load i32, i32* %v18, align 4
  %v27 = mul i32 %v26, %v7
  %v28 = icmp eq i32 %v25, 0
  br i1 %v28, label %b7, label %b6

b6:                                               ; preds = %b10, %b9, %b8, %b5
  %v29 = icmp eq i32 %v21, 0
  br i1 %v29, label %b14, label %b5

b7:                                               ; preds = %b5
  %v30 = icmp ugt i32 %v27, 1
  br i1 %v30, label %b8, label %b9

b8:                                               ; preds = %b7
  %v31 = icmp ugt i32 %v27, 3
  br i1 %v31, label %b10, label %b6

b9:                                               ; preds = %b7
  %v32 = load volatile i32, i32* %v24, align 4
  store volatile i32 %v32, i32* %v24, align 4
  br label %b6

b10:                                              ; preds = %b10, %b8
  %v33 = phi i32 [ %v37, %b10 ], [ %v27, %b8 ]
  %v34 = phi i32* [ %v35, %b10 ], [ %v24, %b8 ]
  %v35 = getelementptr inbounds i32, i32* %v34, i32 -1
  %v36 = load volatile i32, i32* %v34, align 4
  %v37 = add i32 %v33, -4
  %v38 = icmp ugt i32 %v37, 3
  br i1 %v38, label %b10, label %b6

b11:                                              ; preds = %b12, %b2
  %v39 = phi i32 [ %v5, %b2 ], [ %v40, %b12 ]
  %v40 = add i32 %v39, -1
  br i1 %v17, label %b13, label %b12

b12:                                              ; preds = %b13, %b11
  %v41 = icmp eq i32 %v40, 0
  br i1 %v41, label %b14, label %b11

b13:                                              ; preds = %b11
  %v42 = load volatile i32, i32* %v15, align 4
  %v43 = load volatile i32, i32* %v15, align 4
  %v44 = and i32 %v43, %v42
  store volatile i32 %v44, i32* %v15, align 4
  br label %b12

b14:                                              ; preds = %b12, %b6, %b3, %b1
  ret void
}
