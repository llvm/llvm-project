; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: f0
; CHECK-NOT: combine(#0
; CHECK: jump f1

define void @f0(i32* nocapture %a0) #0 {
b0:
  %v0 = load i32, i32* %a0, align 4
  %v1 = zext i32 %v0 to i64
  %v2 = getelementptr inbounds i32, i32* %a0, i32 1
  %v3 = load i32, i32* %v2, align 4
  %v4 = zext i32 %v3 to i64
  %v5 = shl nuw i64 %v4, 32
  %v6 = or i64 %v5, %v1
  tail call void @f1(i64 %v6) #0
  ret void
}

declare void @f1(i64) #0

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
