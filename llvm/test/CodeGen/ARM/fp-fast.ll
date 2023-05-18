; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 -mattr=+vfp4 -enable-unsafe-fp-math %s -o - \
; RUN:  | FileCheck %s

; CHECK: test1
define float @test1(float %x) {
; CHECK-NOT: vfma
; CHECK: vmul.f32
; CHECK-NOT: vfma
  %t1 = fmul float %x, 3.0
  %t2 = call float @llvm.fma.f32(float %x, float 2.0, float %t1)
  ret float %t2
}

; CHECK: test2
define float @test2(float %x, float %y) {
; CHECK-NOT: vmul
; CHECK: vfma.f32
; CHECK-NOT: vmul
  %t1 = fmul float %x, 3.0
  %t2 = call float @llvm.fma.f32(float %t1, float 2.0, float %y)
  ret float %t2
}

; CHECK: test3
define float @test3(float %x, float %y) {
; CHECK-NOT: vfma
; CHECK: vadd.f32
; CHECK-NOT: vfma
  %t2 = call float @llvm.fma.f32(float %x, float 1.0, float %y)
  ret float %t2
}

; CHECK: test4
define float @test4(float %x, float %y) {
; CHECK-NOT: vfma
; CHECK: vsub.f32
; CHECK-NOT: vfma
  %t2 = call float @llvm.fma.f32(float %x, float -1.0, float %y)
  ret float %t2
}

; CHECK: test5
define float @test5(float %x) {
; CHECK-NOT: vfma
; CHECK: vmul.f32
; CHECK-NOT: vfma
  %t2 = call float @llvm.fma.f32(float %x, float 2.0, float %x)
  ret float %t2
}

; CHECK: test6
define float @test6(float %x) {
; CHECK-NOT: vfma
; CHECK: vmul.f32
; CHECK-NOT: vfma
  %t1 = fsub float -0.0, %x
  %t2 = call float @llvm.fma.f32(float %x, float 5.0, float %t1)
  ret float %t2
}

declare float @llvm.fma.f32(float, float, float)
