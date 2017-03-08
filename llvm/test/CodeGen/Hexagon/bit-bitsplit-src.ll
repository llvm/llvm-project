; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; This used to crash. Check for some sane output.
; CHECK: call printf

target triple = "hexagon"

@g0 = external local_unnamed_addr global [4 x i64], align 8
@g1 = external hidden unnamed_addr constant [29 x i8], align 1
@g2 = external hidden unnamed_addr constant [29 x i8], align 1

define void @fred() local_unnamed_addr #0 {
b0:
  %v1 = load i64, i64* getelementptr inbounds ([4 x i64], [4 x i64]* @g0, i32 0, i32 0), align 8
  %v2 = trunc i64 %v1 to i32
  %v3 = lshr i64 %v1, 16
  %v4 = trunc i64 %v3 to i32
  %v5 = and i32 %v4, 255
  %v6 = add nuw nsw i32 0, %v5
  %v7 = add nuw nsw i32 %v6, 0
  %v8 = zext i32 %v7 to i64
  %v9 = and i32 %v2, 65535
  %v10 = and i32 %v4, 65535
  %v11 = add nuw nsw i32 %v10, %v9
  %v12 = zext i32 %v11 to i64
  tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @g1, i32 0, i32 0), i64 %v8) #0
  tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @g2, i32 0, i32 0), i64 %v12) #0
  ret void
}

; Function Attrs: nounwind
declare void @printf(i8* nocapture readonly, ...) local_unnamed_addr #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-hvx-double,-long-calls" }
