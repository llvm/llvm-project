; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that store is post-incremented.
; CHECK: memd(r{{[0-9]+}}++#8) = r

; Function Attrs: nounwind
define void @f0(i32 %a0, i16* nocapture %a1, i16 signext %a2) #0 {
b0:
  %v0 = icmp eq i32 %a0, 0
  br i1 %v0, label %b2, label %b3

b1:                                               ; preds = %b10
  br label %b2

b2:                                               ; preds = %b7, %b1, %b0
  ret void

b3:                                               ; preds = %b0
  %v1 = icmp sgt i32 %a0, 3
  br i1 %v1, label %b4, label %b7

b4:                                               ; preds = %b3
  %v2 = add i32 %a0, -1
  %v3 = and i32 %v2, -4
  %v4 = icmp sgt i32 %v3, 0
  br i1 %v4, label %b5, label %b7

b5:                                               ; preds = %b4
  %v5 = insertelement <4 x i16> undef, i16 %a2, i32 0
  %v6 = insertelement <4 x i16> %v5, i16 %a2, i32 1
  %v7 = insertelement <4 x i16> %v6, i16 %a2, i32 2
  %v8 = insertelement <4 x i16> %v7, i16 %a2, i32 3
  br label %b9

b6:                                               ; preds = %b9
  br label %b7

b7:                                               ; preds = %b6, %b4, %b3
  %v9 = phi i32 [ 0, %b3 ], [ %v3, %b4 ], [ %v3, %b6 ]
  %v10 = icmp slt i32 %v9, %a0
  br i1 %v10, label %b8, label %b2

b8:                                               ; preds = %b7
  br label %b10

b9:                                               ; preds = %b9, %b5
  %v11 = phi i32 [ 0, %b5 ], [ %v12, %b9 ]
  %v12 = add nsw i32 %v11, 4
  %v13 = getelementptr i16, i16* %a1, i32 %v11
  %v14 = bitcast i16* %v13 to <4 x i16>*
  %v15 = load <4 x i16>, <4 x i16>* %v14, align 16
  %v16 = add <4 x i16> %v15, %v8
  store <4 x i16> %v16, <4 x i16>* %v14, align 16
  %v17 = icmp slt i32 %v12, %v3
  br i1 %v17, label %b9, label %b6

b10:                                              ; preds = %b10, %b8
  %v18 = phi i32 [ %v19, %b10 ], [ %v9, %b8 ]
  %v19 = add nsw i32 %v18, 1
  %v20 = getelementptr i16, i16* %a1, i32 %v18
  %v21 = load i16, i16* %v20, align 2
  %v22 = add i16 %v21, %a2
  store i16 %v22, i16* %v20, align 2
  %v23 = icmp eq i32 %v19, %a0
  br i1 %v23, label %b1, label %b10
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
