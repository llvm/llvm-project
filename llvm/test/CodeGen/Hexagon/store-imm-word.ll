; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: memw{{.*}} = #-1

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  store i32 -1, ptr %a0, align 4
  ret void
}

attributes #0 = { nounwind }
