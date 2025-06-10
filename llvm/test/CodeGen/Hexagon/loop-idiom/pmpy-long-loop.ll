; RUN: opt -hexagon-loop-idiom -S < %s | FileCheck %s
; RUN: opt -p hexagon-loop-idiom -S < %s | FileCheck %s
;
; The number of nested selects caused the simplification loop to take
; more than the maximum number of iterations. This caused the compiler
; to crash under suspicion of an infinite loop. This (still reduced)
; testcase shows a legitimate case where this limit was exceeded.
; Instead of crashing, gracefully abort the simplification.
;
; Check for sane output.
; CHECK: define void @fred

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred() unnamed_addr #0 {
b0:
  %v1 = select i1 false, i32 undef, i32 2
  br label %b2

b2:                                               ; preds = %b2, %b0
  %v3 = sext i16 undef to i32
  %v4 = add nsw i32 %v1, %v3
  %v5 = select i1 undef, i32 undef, i32 %v4
  %v6 = icmp slt i32 %v5, undef
  %v7 = select i1 %v6, i32 %v5, i32 undef
  %v8 = icmp slt i32 %v7, 0
  %v9 = select i1 %v8, i32 %v7, i32 0
  %v10 = sub i32 undef, undef
  %v11 = add i32 %v10, %v9
  %v12 = sext i16 undef to i32
  %v13 = sext i16 undef to i32
  %v14 = add nsw i32 %v1, %v13
  %v15 = select i1 undef, i32 undef, i32 %v14
  %v16 = icmp slt i32 %v15, undef
  %v17 = select i1 %v16, i32 %v15, i32 undef
  %v18 = select i1 undef, i32 %v17, i32 %v12
  %v19 = add i32 undef, %v18
  %v20 = sext i16 undef to i32
  %v21 = sext i16 0 to i32
  %v22 = add nsw i32 %v1, %v21
  %v23 = sext i16 undef to i32
  %v24 = add nsw i32 %v1, %v23
  %v25 = select i1 undef, i32 undef, i32 %v24
  %v26 = icmp slt i32 %v25, %v22
  %v27 = select i1 %v26, i32 %v25, i32 %v22
  %v28 = icmp slt i32 %v27, %v20
  %v29 = select i1 %v28, i32 %v27, i32 %v20
  %v30 = add i32 undef, %v29
  %v31 = add i32 %v11, undef
  %v32 = add i32 %v31, undef
  %v33 = add i32 %v32, %v19
  %v34 = add i32 %v33, %v30
  %v35 = add nsw i32 %v34, 32768
  %v36 = icmp ult i32 %v35, 65536
  %v37 = select i1 %v36, i32 %v34, i32 undef
  br i1 undef, label %b2, label %b38

b38:                                              ; preds = %b2
  unreachable
}

attributes #0 = { "target-cpu"="hexagonv60" }
