; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: memh{{.*}} = #-1

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i16* %a0) #0 {
b0:
  store i16 -1, i16* %a0, align 2
  ret void
}

attributes #0 = { nounwind }
