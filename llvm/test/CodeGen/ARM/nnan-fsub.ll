; RUN: llc -mcpu=cortex-a9 < %s | FileCheck %s

target triple = "armv7-apple-ios"

; CHECK-LABEL: test
define float @test(float %x, float %y) {
entry:
; CHECK: vmul.f32
; CHECK-NEXT: vsub.f32
  %0 = fmul float %x, %y
  %1 = fsub float %0, %0
  ret float %1
}

; CHECK-LABEL: test_nnan
define float @test_nnan(float %x, float %y) {
entry:
; CHECK: mov r0, #0
  %0 = fmul float %x, %y
  %1 = fsub nnan float %0, %0
  ret float %1
}
