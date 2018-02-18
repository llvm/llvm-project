; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - | FileCheck %s --check-prefix=CHECK-VFP
; RUN: llc -mtriple=thumbv7-eabi -mattr=+vfp2 %s -o - | FileCheck %s --check-prefix=CHECK-VFP

declare float @llvm.arm.vcvtr.f32(float)
declare float @llvm.arm.vcvtru.f32(float)
declare float @llvm.arm.vcvtr.f64(double)
declare float @llvm.arm.vcvtru.f64(double)

define float @test_vcvtrf0(float %f) {
entry:
; CHECK-VFP:  vcvtr.s32.f32  s0, s0
  %vcvtr = tail call float @llvm.arm.vcvtr.f32(float %f)
  ret float %vcvtr
}

define float @test_vcvtrf1(float %f) {
entry:
; CHECK-VFP:  vcvtr.u32.f32  s0, s0
  %vcvtr = tail call float @llvm.arm.vcvtru.f32(float %f)
  ret float %vcvtr
}

define float @test_vcvtrd0(double %d) {
entry:
; CHECK-VFP: vcvtr.s32.f64  s0, d{{.*}}
  %vcvtr = tail call float @llvm.arm.vcvtr.f64(double %d)
  ret float %vcvtr
}

define float @test_vcvtrd1(double %d) {
entry:
; CHECK-VFP: vcvtr.u32.f64  s0, d{{.*}}
  %vcvtr = tail call float @llvm.arm.vcvtru.f64(double %d)
  ret float %vcvtr
}
