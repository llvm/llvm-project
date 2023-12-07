; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v2 = load i64, ptr %v0, align 8
; CHECK: 	call f1
  %v3 = call i32 @f1(i64 %v2)
  unreachable
}

; Function Attrs: inlinehint nounwind
declare i32 @f1(i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { inlinehint nounwind }
