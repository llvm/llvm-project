; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

@g0 = common global [16 x i32] zeroinitializer, align 8
@g1 = internal global [16 x i32] zeroinitializer, align 8

; CHECK-NOT: g1.*lcomm

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  call void @f1(i32* getelementptr inbounds ([16 x i32], [16 x i32]* @g0, i32 0, i32 0), i32* getelementptr inbounds ([16 x i32], [16 x i32]* @g1, i32 0, i32 0))
  %v0 = getelementptr inbounds [16 x i32], [16 x i32]* @g0, i32 0, i32 %a0
  %v1 = load i32, i32* %v0, align 4
  %v2 = getelementptr inbounds [16 x i32], [16 x i32]* @g1, i32 0, i32 %a0
  %v3 = load i32, i32* %v2, align 4
  %v4 = add nsw i32 %v1, %v3
  ret i32 %v4
}

declare void @f1(i32*, i32*)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
