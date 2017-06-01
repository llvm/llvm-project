; RUN: llc < %s -mtriple=arm64-eabi -mcpu=generic -aarch64-neon-syntax=apple | FileCheck %s

define void @test0f(float* nocapture %x, float %a) #0 {
entry:
  %0 = insertelement <4 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %a, i32 0
  %1 = bitcast float* %x to <4 x float>*
  store <4 x float> %0, <4 x float>* %1, align 16
  ret void

  ; CHECK-LABEL: test0f
  ; CHECK: movi.2d v[[TEMP:[0-9]+]], #0000000000000000
  ; CHECK: ins.s v[[TEMP]][0], v{{[0-9]+}}[0]
  ; CHECK: str q[[TEMP]], [x0]
  ; CHECK: ret


}


define void @test1f(float* nocapture %x, float %a) #0 {
entry:
  %0 = insertelement <4 x float> <float undef, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, float %a, i32 0
  %1 = bitcast float* %x to <4 x float>*
  store <4 x float> %0, <4 x float>* %1, align 16
  ret void

  ; CHECK-LABEL: test1f
  ; CHECK: fmov  s[[TEMP:[0-9]+]], #1.0000000
  ; CHECK: dup.4s  v[[TEMP2:[0-9]+]], v[[TEMP]][0]
  ; CHECK: ins.s v[[TEMP2]][0], v0[0]
  ; CHECK: str q[[TEMP2]], [x0]
  ; CHECK: ret
}
