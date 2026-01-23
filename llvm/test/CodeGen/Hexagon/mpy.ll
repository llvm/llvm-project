; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: += mpyi

define void @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store i32 %a0, ptr %v0, align 4
  store i32 %a1, ptr %v1, align 4
  store i32 %a2, ptr %v2, align 4
  %v3 = load i32, ptr %v1, align 4
  %v4 = load i32, ptr %v0, align 4
  %v5 = mul nsw i32 %v3, %v4
  %v6 = load i32, ptr %v2, align 4
  %v7 = add nsw i32 %v5, %v6
  store i32 %v7, ptr %v1, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
