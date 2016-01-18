; RUN: opt < %s -instcombine -S | FileCheck %s

define float @mytan(float %x) #0 {
entry:
  %call = call float @atanf(float %x)
  %call1 = call float @tanf(float %call)
  ret float %call1
}

; CHECK-LABEL: define float @mytan(
; CHECK:   ret float %x

declare float @tanf(float) #0
declare float @atanf(float) #0
attributes #0 = { "unsafe-fp-math"="true" }
