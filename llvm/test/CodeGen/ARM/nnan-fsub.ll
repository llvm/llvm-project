; RUN: llc -mcpu=cortex-a9 < %s | FileCheck -check-prefix=SAFE %s
; RUN: llc -mcpu=cortex-a9 --enable-no-nans-fp-math < %s | FileCheck -check-prefix=FAST %s

target triple = "armv7-apple-ios"

; SAFE: test
; FAST: test
define float @test(float %x, float %y) {
entry:
; SAFE: vmul.f32
; SAFE: vsub.f32
; FAST: mov r0, #0
  %0 = fmul float %x, %y
  %1 = fsub float %0, %0
  ret float %1
}


