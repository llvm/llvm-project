; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define float @mytan(float %x) {
  %call = call fast float @atanf(float %x)
  %call1 = call fast float @tanf(float %call)
  ret float %call1
}

; CHECK-LABEL: define float @mytan(
; CHECK:   ret float %x

define float @test2(float ()* %fptr) {
  %call1 = call fast float %fptr()
  %tan = call fast float @tanf(float %call1)
  ret float %tan
}

; CHECK-LABEL: @test2
; CHECK: tanf

declare float @tanf(float)
declare float @atanf(float)

