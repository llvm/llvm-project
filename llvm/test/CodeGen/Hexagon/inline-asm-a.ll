; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that constraint a is handled correctly.
; CHECK: [[M:m[01]]] = r1
; CHECK: memw(r0++[[M]]) = r2

target triple = "hexagon"

; Function Attrs: nounwind
define void @foo(ptr %a, i32 %m, i32 %v) #0 {
entry:
  tail call void asm sideeffect "memw($0++$1) = $2", "r,a,r,~{memory}"(ptr %a, i32 %m, i32 %v)
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
