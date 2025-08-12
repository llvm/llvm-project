; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that this testcase compiles successfully.
; CHECK: allocframe

target triple = "hexagon"

@g0 = external global [1024 x i8], align 8
@g1 = external global [1024 x i8], align 8
@g2 = external global [1024 x i8], align 8
@g3 = external global [1024 x i8], align 8
@g4 = external hidden unnamed_addr constant [40 x i8], align 1

; Function Attrs: nounwind
define void @fred() local_unnamed_addr #0 {
b0:
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  %v3 = load i8, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 9), align 1
  %v4 = load i8, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 10), align 2
  store i32 24, ptr %v1, align 4
  store i8 %v3, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 16), align 8
  store i8 %v4, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 10), align 2
  store i32 44, ptr %v2, align 4
  store i16 0, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 4), align 4
  %v5 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 11), align 1
  store i16 %v5, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 18), align 2
  %v6 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 13), align 1
  store i32 %v6, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 20), align 4
  %v7 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 17), align 1
  store i16 %v7, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 24), align 8
  %v8 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 23), align 1
  store i16 %v8, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 32), align 8
  %v9 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 25), align 1
  store i32 %v9, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 36), align 4
  %v10 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 29), align 1
  store i16 %v10, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 40), align 8
  %v11 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 31), align 1
  store i32 %v11, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 44), align 4
  %v12 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 35), align 1
  store i16 %v12, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 48), align 8
  %v13 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 37), align 1
  store i32 %v13, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 52), align 4
  %v14 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 41), align 1
  store i16 %v14, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 56), align 8
  %v15 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 43), align 1
  store i32 %v15, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 60), align 4
  %v16 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 47), align 1
  store i16 %v16, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 64), align 8
  %v17 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 49), align 1
  store i32 %v17, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 68), align 4
  %v18 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 53), align 1
  store i16 %v18, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 72), align 8
  %v19 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 55), align 1
  store i32 %v19, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 76), align 4
  %v20 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 61), align 1
  store i32 %v20, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 84), align 4
  %v21 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 73), align 1
  store i32 %v21, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 100), align 4
  store i32 104, ptr %v1, align 4
  store i8 %v4, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 10), align 2
  store i16 %v8, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 23), align 1
  store i32 %v9, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 25), align 1
  store i16 %v10, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 29), align 1
  store i32 %v11, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 31), align 1
  store i16 %v12, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 35), align 1
  store i32 %v13, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 37), align 1
  store i16 %v14, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 41), align 1
  store i32 %v15, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 43), align 1
  store i16 %v16, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 47), align 1
  store i32 %v17, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 49), align 1
  store i32 %v19, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 55), align 1
  store i32 %v20, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 61), align 1
  store i32 %v21, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 73), align 1
  %v22 = trunc i32 %v6 to i8
  store i8 %v22, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 20), align 4
  store i32 24, ptr %v1, align 4
  store i16 0, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 4), align 4
  store i8 %v3, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 9), align 1
  store i16 %v5, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 11), align 1
  store i8 %v22, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 13), align 1
  store i32 14, ptr %v2, align 4
  store i8 %v4, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 17), align 1
  %v23 = load i64, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 11), align 1
  store i64 %v23, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 24), align 8
  %v24 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 19), align 1
  store i16 %v24, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 32), align 8
  %v25 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 21), align 1
  store i32 %v25, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 36), align 4
  %v26 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 25), align 1
  store i32 %v26, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 40), align 8
  %v27 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 29), align 1
  store i16 %v27, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 44), align 4
  %v28 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 31), align 1
  store i16 %v28, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 46), align 2
  %v29 = load i8, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 33), align 1
  store i8 %v29, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 48), align 8
  %v30 = load i8, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 34), align 2
  store i8 %v30, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 56), align 8
  %v31 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 35), align 1
  store i32 %v31, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 60), align 4
  %v32 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 39), align 1
  store i32 72, ptr %v1, align 4
  store i32 0, ptr @g2, align 8
  store i16 0, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 4), align 4
  store i8 %v3, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 9), align 1
  store i32 %v25, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 21), align 1
  store i32 %v26, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 25), align 1
  store i16 %v27, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 29), align 1
  store i16 %v28, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 31), align 1
  store i8 %v29, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 33), align 1
  store i8 %v30, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 34), align 2
  store i32 %v31, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 35), align 1
  store i32 %v32, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 39), align 1
  store i32 43, ptr %v2, align 4
  %v33 = load i8, ptr @g1, align 8
  %v34 = zext i8 %v33 to i32
  tail call void (ptr, ...) @printf(ptr @g4, i32 %v34, i32 0) #0
  %v35 = load i8, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 7), align 1
  store i8 %v35, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 7), align 1
  %v36 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 17), align 1
  store i16 %v36, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 24), align 8
  %v37 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 19), align 1
  %v38 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 31), align 1
  store i32 %v38, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 44), align 4
  %v39 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 35), align 1
  %v40 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 37), align 1
  store i32 %v40, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 52), align 4
  %v41 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 43), align 1
  store i32 %v41, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 60), align 4
  %v42 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 47), align 1
  store i16 %v42, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 64), align 8
  %v43 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 49), align 1
  store i32 %v43, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 68), align 4
  %v44 = load i16, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 59), align 1
  store i16 %v44, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 80), align 8
  %v45 = load i32, ptr getelementptr inbounds ([1024 x i8], ptr @g0, i32 0, i32 67), align 1
  store i32 %v45, ptr getelementptr inbounds ([1024 x i8], ptr @g3, i32 0, i32 92), align 4
  store i32 96, ptr %v1, align 4
  store i8 %v35, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 7), align 1
  store i16 %v36, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 17), align 1
  store i32 %v37, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 19), align 1
  store i32 %v38, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 31), align 1
  store i16 %v39, ptr getelementptr inbounds ([1024 x i8], ptr @g2, i32 0, i32 35), align 1
  call void (ptr, ...) @printf(ptr @g4, i32 0, i32 0) #0
  call void (ptr, ...) @printf(ptr @g4, i32 undef, i32 0) #0
  unreachable
}

declare void @printf(ptr nocapture readonly, ...) local_unnamed_addr #0

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
