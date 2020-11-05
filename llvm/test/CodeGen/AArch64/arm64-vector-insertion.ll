; RUN: llc < %s -mtriple=arm64-eabi -mcpu=generic -aarch64-neon-syntax=apple | FileCheck %s

define void @test0f(float* nocapture %x, float %a) #0 {
entry:
  %0 = insertelement <4 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %a, i32 0
  %1 = bitcast float* %x to <4 x float>*
  store <4 x float> %0, <4 x float>* %1, align 16
  ret void

  ; CHECK-LABEL: test0f
  ; CHECK: movi.2d v[[TEMP:[0-9]+]], #0
  ; CHECK: mov.s v[[TEMP]][0], v{{[0-9]+}}[0]
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
  ; CHECK: fmov.4s v[[TEMP:[0-9]+]], #1.0
  ; CHECK: mov.s v[[TEMP]][0], v0[0]
  ; CHECK: str q[[TEMP]], [x0]
  ; CHECK: ret
}

; TODO: This should jsut be a dup + clearing lane 4.
define <4 x float> @test2(float %a) {
; CHECK-LABEL: test2:
; CHECK:       bb.0:
; CHECK-NEXT:    movi.2d v1, #0000000000000000
; CHECK-NEXT:    // kill
; CHECK-NEXT:    mov.s  v1[0], v0[0]
; CHECK-NEXT:    mov.s   v1[1], v0[0]
; CHECK-NEXT:    mov.s   v1[2], v0[0]
; CHECK-NEXT:    mov.16b v0, v1
; CHECK-NEXT:   ret
;
entry:
  %0 = insertelement <4 x float> <float undef, float undef, float undef, float 0.000000e+00>, float %a, i32 0
  %1 = insertelement <4 x float> %0, float %a, i32 1
  %vecinit3 = insertelement <4 x float> %1, float %a, i32 2
  ret <4 x float> %vecinit3
}

; TODO: This should jsut be a mov.s v0[3], wzr
define <4 x float> @test3(<4 x float> %a) #0 {
; CHECK-LABEL: test3:
; CHECK:       bb.0:
; CHECK-NEXT:    fmov    s1, wzr
; CHECK-NEXT:    mov.s   v0[3], v1[0]
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <4 x float> %a, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit5
}
