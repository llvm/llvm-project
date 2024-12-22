; RUN: llc -march=hexagon < %s | FileCheck %s
; This testcase used to fail with "cannot select 'i1 = add x, y'".
; Check for some sane output:
; CHECK: xor(p{{[0-3]}},p{{[0-3]}})

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @foo(ptr nocapture %a0) local_unnamed_addr #0 {
b1:
  %v2 = getelementptr inbounds i32, ptr %a0, i32 26
  %v3 = load i32, ptr %v2, align 4
  %v4 = add nsw i32 %v3, 1
  %v5 = load i32, ptr %a0, align 4
  br label %b6

b6:                                               ; preds = %b28, %b1
  %v7 = phi i32 [ %v29, %b28 ], [ %v5, %b1 ]
  %v8 = mul nsw i32 %v4, %v7
  %v9 = add nsw i32 %v8, %v7
  %v10 = mul i32 %v7, %v7
  %v11 = mul i32 %v10, %v9
  %v12 = add nsw i32 %v11, 1
  %v13 = mul nsw i32 %v12, %v7
  %v14 = add nsw i32 %v13, %v7
  %v15 = mul i32 %v10, %v14
  %v16 = and i32 %v15, 1
  %v17 = add nsw i32 %v16, -1
  %v18 = mul i32 %v10, %v7
  %v19 = mul i32 %v18, %v11
  %v20 = mul i32 %v19, %v17
  %v21 = and i32 %v20, 1
  %v22 = add nsw i32 %v21, -1
  %v23 = mul nsw i32 %v22, %v3
  %v24 = sub nsw i32 %v7, %v23
  %v25 = mul i32 %v10, %v24
  %v26 = sub i32 0, %v7
  %v27 = icmp eq i32 %v25, %v26
  br i1 %v27, label %b30, label %b28

b28:                                              ; preds = %b6
  %v29 = add nsw i32 %v3, %v7
  store i32 %v29, ptr %a0, align 4
  br label %b6

b30:                                              ; preds = %b6
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" }
