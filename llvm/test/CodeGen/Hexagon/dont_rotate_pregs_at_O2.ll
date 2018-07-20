; RUN: llc -O2 -march=hexagon -hexagon-eif=0 < %s | FileCheck %s

; Make sure we are not rotating registers at O2.
; CHECK-NOT: p1 =
; CHECK-NOT: p2 =

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5) #0 {
b0:
  %v0 = icmp slt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = mul nsw i32 %a1, %a0
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ %v1, %b1 ], [ 0, %b0 ]
  %v3 = icmp sgt i32 %a0, %a1
  br i1 %v3, label %b3, label %b4

b3:                                               ; preds = %b2
  %v4 = mul nsw i32 %a2, %a1
  %v5 = add nsw i32 %v4, %a0
  %v6 = add nsw i32 %v5, %a3
  %v7 = add nsw i32 %v6, %v2
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v8 = phi i32 [ %v7, %b3 ], [ %v2, %b2 ]
  %v9 = icmp sgt i32 %a2, %a3
  br i1 %v9, label %b5, label %b6

b5:                                               ; preds = %b4
  %v10 = mul nsw i32 %a3, %a2
  %v11 = add nsw i32 %v8, %v10
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v12 = phi i32 [ %v11, %b5 ], [ %v8, %b4 ]
  %v13 = icmp sgt i32 %a3, %a2
  br i1 %v13, label %b7, label %b8

b7:                                               ; preds = %b6
  %v14 = sdiv i32 %a3, 2
  %v15 = mul nsw i32 %v14, %a0
  %v16 = add nsw i32 %v15, %v12
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v17 = phi i32 [ %v16, %b7 ], [ %v12, %b6 ]
  %v18 = icmp slt i32 %a4, %a5
  br i1 %v18, label %b9, label %b10

b9:                                               ; preds = %b8
  %v19 = mul i32 %a4, %a3
  %v20 = mul i32 %v19, %a5
  %v21 = add nsw i32 %v17, %v20
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v22 = phi i32 [ %v21, %b9 ], [ %v17, %b8 ]
  ret i32 %v22
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv55" }
