; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: f0
; CHECK-NOT: allocframe
; CHECK-NOT: memd(r29
; CHECK: jump f1

define void @f0(i32 %a0) #0 {
b0:
  %v0 = add nsw i32 %a0, 3
  %v1 = tail call i32 @f1(i32 %v0) #0
  ret void
}

declare i32 @f1(...) #0

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
