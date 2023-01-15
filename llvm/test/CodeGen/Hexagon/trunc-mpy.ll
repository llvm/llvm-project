; RUN: llc -march=hexagon -disable-hexagon-peephole < %s  | FileCheck %s

; Test that we're generating a 32-bit multiply high instead of a 64-bit version,
; when using the high 32-bits only.

; CHECK-LABEL: f0:
; CHECK-NOT:  r{{[0-9]+}}:{{[0-9]+}} = mpy(
define void @f0(ptr nocapture readonly %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = getelementptr i32, ptr %a1, i32 448
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  %v1 = getelementptr inbounds i32, ptr %a0, i32 64
  %v2 = load i32, ptr %a0, align 4
  %v3 = getelementptr inbounds i32, ptr %a0, i32 2
  %v4 = load i32, ptr %v1, align 4
  %v5 = sext i32 %v2 to i64
  %v6 = sext i32 %v4 to i64
  %v7 = mul nsw i64 %v6, %v5
  %v8 = lshr i64 %v7, 32
  %v9 = trunc i64 %v8 to i32
  %v10 = sub nsw i32 0, %v9
  %v11 = getelementptr inbounds i32, ptr %v0, i32 1
  store i32 %v10, ptr %v1, align 4
  ret void
}

; Similar to above, but using the operands of the multiply are expressions.

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]+}} = mpy(
define void @f1(i32 %a0, i32 %a1, ptr nocapture readonly %a2, ptr nocapture %a3) #0 {
b0:
  %v0 = getelementptr i32, ptr %a3, i32 448
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  %v1 = getelementptr inbounds i32, ptr %a2, i32 64
  %v2 = sext i32 %a0 to i64
  %v3 = sext i32 %a1 to i64
  %v4 = mul nsw i64 %v3, %v2
  %v5 = lshr i64 %v4, 32
  %v6 = trunc i64 %v5 to i32
  %v7 = sub nsw i32 0, %v6
  %v8 = getelementptr inbounds i32, ptr %v0, i32 1
  store i32 %v7, ptr %v1, align 4
  ret void
}

; Check that the transform occurs when the loads can be post-incremented.

; CHECK-LABEL: f2:
; CHECK: r{{[0-9]+}} = mpy(
define void @f2(ptr nocapture readonly %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = getelementptr i32, ptr %a1, i32 448
  br label %b1

b1:                                               ; preds = %b0
  %v1 = getelementptr inbounds i32, ptr %a0, i32 64
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v2 = phi ptr [ %v0, %b1 ], [ %v14, %b2 ]
  %v3 = phi ptr [ %v1, %b1 ], [ undef, %b2 ]
  %v4 = phi ptr [ null, %b1 ], [ %v6, %b2 ]
  %v5 = load i32, ptr %v4, align 4
  %v6 = getelementptr inbounds i32, ptr %v4, i32 2
  %v7 = load i32, ptr %v3, align 4
  %v8 = sext i32 %v5 to i64
  %v9 = sext i32 %v7 to i64
  %v10 = mul nsw i64 %v9, %v8
  %v11 = lshr i64 %v10, 32
  %v12 = trunc i64 %v11 to i32
  %v13 = sub nsw i32 0, %v12
  %v14 = getelementptr inbounds i32, ptr %v2, i32 1
  store i32 %v13, ptr %v2, align 4
  br label %b2
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
