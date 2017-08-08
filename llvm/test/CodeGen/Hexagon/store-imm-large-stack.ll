; RUN: llc -march=hexagon < %s | FileCheck %s
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
  %v3 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 9), align 1
  %v4 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 10), align 2
  store i32 24, i32* %v1, align 4
  store i8 %v3, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 16), align 8
  store i8 %v4, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 10), align 2
  store i32 44, i32* %v2, align 4
  store i16 0, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 4) to i16*), align 4
  %v5 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 11) to i16*), align 1
  store i16 %v5, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 18) to i16*), align 2
  %v6 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 13) to i32*), align 1
  store i32 %v6, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 20) to i32*), align 4
  %v7 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 17) to i16*), align 1
  store i16 %v7, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 24) to i16*), align 8
  %v8 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 23) to i16*), align 1
  store i16 %v8, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 32) to i16*), align 8
  %v9 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 25) to i32*), align 1
  store i32 %v9, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 36) to i32*), align 4
  %v10 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 29) to i16*), align 1
  store i16 %v10, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 40) to i16*), align 8
  %v11 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 31) to i32*), align 1
  store i32 %v11, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 44) to i32*), align 4
  %v12 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 35) to i16*), align 1
  store i16 %v12, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 48) to i16*), align 8
  %v13 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 37) to i32*), align 1
  store i32 %v13, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 52) to i32*), align 4
  %v14 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 41) to i16*), align 1
  store i16 %v14, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 56) to i16*), align 8
  %v15 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 43) to i32*), align 1
  store i32 %v15, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 60) to i32*), align 4
  %v16 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 47) to i16*), align 1
  store i16 %v16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 64) to i16*), align 8
  %v17 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 49) to i32*), align 1
  store i32 %v17, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 68) to i32*), align 4
  %v18 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 53) to i16*), align 1
  store i16 %v18, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 72) to i16*), align 8
  %v19 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 55) to i32*), align 1
  store i32 %v19, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 76) to i32*), align 4
  %v20 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 61) to i32*), align 1
  store i32 %v20, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 84) to i32*), align 4
  %v21 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 73) to i32*), align 1
  store i32 %v21, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 100) to i32*), align 4
  store i32 104, i32* %v1, align 4
  store i8 %v4, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 10), align 2
  store i16 %v8, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 23) to i16*), align 1
  store i32 %v9, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 25) to i32*), align 1
  store i16 %v10, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 29) to i16*), align 1
  store i32 %v11, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 31) to i32*), align 1
  store i16 %v12, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 35) to i16*), align 1
  store i32 %v13, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 37) to i32*), align 1
  store i16 %v14, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 41) to i16*), align 1
  store i32 %v15, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 43) to i32*), align 1
  store i16 %v16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 47) to i16*), align 1
  store i32 %v17, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 49) to i32*), align 1
  store i32 %v19, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 55) to i32*), align 1
  store i32 %v20, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 61) to i32*), align 1
  store i32 %v21, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 73) to i32*), align 1
  %v22 = trunc i32 %v6 to i8
  store i8 %v22, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 20), align 4
  store i32 24, i32* %v1, align 4
  store i16 0, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 4) to i16*), align 4
  store i8 %v3, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 9), align 1
  store i16 %v5, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 11) to i16*), align 1
  store i8 %v22, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 13), align 1
  store i32 14, i32* %v2, align 4
  store i8 %v4, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 17), align 1
  %v23 = load i64, i64* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 11) to i64*), align 1
  store i64 %v23, i64* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 24) to i64*), align 8
  %v24 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 19) to i16*), align 1
  store i16 %v24, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 32) to i16*), align 8
  %v25 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 21) to i32*), align 1
  store i32 %v25, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 36) to i32*), align 4
  %v26 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 25) to i32*), align 1
  store i32 %v26, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 40) to i32*), align 8
  %v27 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 29) to i16*), align 1
  store i16 %v27, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 44) to i16*), align 4
  %v28 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 31) to i16*), align 1
  store i16 %v28, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 46) to i16*), align 2
  %v29 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 33), align 1
  store i8 %v29, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 48), align 8
  %v30 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 34), align 2
  store i8 %v30, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 56), align 8
  %v31 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 35) to i32*), align 1
  store i32 %v31, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 60) to i32*), align 4
  %v32 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 39) to i32*), align 1
  store i32 72, i32* %v1, align 4
  store i32 0, i32* bitcast ([1024 x i8]* @g2 to i32*), align 8
  store i16 0, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 4) to i16*), align 4
  store i8 %v3, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 9), align 1
  store i32 %v25, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 21) to i32*), align 1
  store i32 %v26, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 25) to i32*), align 1
  store i16 %v27, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 29) to i16*), align 1
  store i16 %v28, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 31) to i16*), align 1
  store i8 %v29, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 33), align 1
  store i8 %v30, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 34), align 2
  store i32 %v31, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 35) to i32*), align 1
  store i32 %v32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 39) to i32*), align 1
  store i32 43, i32* %v2, align 4
  %v33 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g1, i32 0, i32 0), align 8
  %v34 = zext i8 %v33 to i32
  tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @g4, i32 0, i32 0), i32 %v34, i32 0) #0
  %v35 = load i8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 7), align 1
  store i8 %v35, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 7), align 1
  %v36 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 17) to i16*), align 1
  store i16 %v36, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 24) to i16*), align 8
  %v37 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 19) to i32*), align 1
  %v38 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 31) to i32*), align 1
  store i32 %v38, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 44) to i32*), align 4
  %v39 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 35) to i16*), align 1
  %v40 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 37) to i32*), align 1
  store i32 %v40, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 52) to i32*), align 4
  %v41 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 43) to i32*), align 1
  store i32 %v41, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 60) to i32*), align 4
  %v42 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 47) to i16*), align 1
  store i16 %v42, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 64) to i16*), align 8
  %v43 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 49) to i32*), align 1
  store i32 %v43, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 68) to i32*), align 4
  %v44 = load i16, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 59) to i16*), align 1
  store i16 %v44, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 80) to i16*), align 8
  %v45 = load i32, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g0, i32 0, i32 67) to i32*), align 1
  store i32 %v45, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g3, i32 0, i32 92) to i32*), align 4
  store i32 96, i32* %v1, align 4
  store i8 %v35, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 7), align 1
  store i16 %v36, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 17) to i16*), align 1
  store i32 %v37, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 19) to i32*), align 1
  store i32 %v38, i32* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 31) to i32*), align 1
  store i16 %v39, i16* bitcast (i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @g2, i32 0, i32 35) to i16*), align 1
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @g4, i32 0, i32 0), i32 0, i32 0) #0
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @g4, i32 0, i32 0), i32 undef, i32 0) #0
  unreachable
}

declare void @printf(i8* nocapture readonly, ...) local_unnamed_addr #0

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
