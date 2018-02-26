; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Make sure we can handle the 'q' constraint in the 128-byte mode.

target triple = "hexagon"

; CHECK-LABEL: fred
; CHECK: if (q{{[0-3]}}) vmem
define void @fred() #0 {
  tail call void asm sideeffect "if ($0) vmem($1) = $2;", "q,r,v,~{memory}"(<32 x i32> undef, <32 x i32>* undef, <32 x i32> undef) #0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
