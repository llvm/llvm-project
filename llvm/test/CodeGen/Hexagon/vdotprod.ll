; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate a single packet for the vectorized dot product loop.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK: mpyi
; CHECK: mpyi
; CHECK: memd
; CHECK: memd
; CHECK-NOT: {
; CHECK: }{{[ \t]*}}:endloop

; Function Attrs: nounwind readonly
define i32 @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi ptr [ %a0, %b0 ], [ %v21, %b1 ]
  %v3 = phi ptr [ %a1, %b0 ], [ %v22, %b1 ]
  %v4 = phi i32 [ 0, %b0 ], [ %v19, %b1 ]
  %v5 = phi i32 [ 0, %b0 ], [ %v14, %b1 ]
  %v6 = phi i32 [ 0, %b0 ], [ %v18, %b1 ]
  %v7 = load i64, ptr %v2, align 8
  %v8 = trunc i64 %v7 to i32
  %v9 = lshr i64 %v7, 32
  %v10 = load i64, ptr %v3, align 8
  %v11 = trunc i64 %v10 to i32
  %v12 = lshr i64 %v10, 32
  %v13 = mul nsw i32 %v11, %v8
  %v14 = add nsw i32 %v13, %v5
  %v15 = trunc i64 %v9 to i32
  %v16 = trunc i64 %v12 to i32
  %v17 = mul nsw i32 %v16, %v15
  %v18 = add nsw i32 %v17, %v6
  %v19 = add nsw i32 %v4, 1
  %v20 = icmp eq i32 %v19, 199
  %v21 = getelementptr i64, ptr %v2, i32 1
  %v22 = getelementptr i64, ptr %v3, i32 1
  br i1 %v20, label %b2, label %b1

b2:                                               ; preds = %b1
  %v23 = add nsw i32 %v14, %v18
  ret i32 %v23
}

attributes #0 = { nounwind readonly }
