; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: .lcomm  g0,4,4,4

target triple = "hexagon"

@g0 = internal global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = load i32, ptr @g0, align 4
  ret i32 %v0
}

attributes #0 = { nounwind }
