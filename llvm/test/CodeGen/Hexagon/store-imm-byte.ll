; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: memb{{.*}} = #-1

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  store i8 -1, ptr %a0, align 2
  ret void
}

attributes #0 = { nounwind }
