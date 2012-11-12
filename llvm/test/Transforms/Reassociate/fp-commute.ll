; RUN: opt -reassociate -S < %s | FileCheck %s

target triple = "armv7-apple-ios"

declare void @use(float)

; CHECK: test
define void @test(float %x, float %y) {
entry:
; CHECK: fmul float %x, %y
; CHECK: fmul float %x, %y
  %0 = fmul float %x, %y
  %1 = fmul float %y, %x
  %2 = fsub float %0, %1
  call void @use(float %0)
  call void @use(float %2)
  ret void
}
