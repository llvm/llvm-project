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
define <4 x float> @test_insert_3_f32_undef_zero_vector(float %a) {
; CHECK-LABEL: test_insert_3_f32_undef_zero_vector:
; CHECK:       bb.0:
; CHECK-NEXT:   movi.2d v1, #0000000000000000
; CHECK-NEXT:   // kill
; CHECK-NEXT:   mov.s  v1[0], v0[0]
; CHECK-NEXT:   mov.s  v1[1], v0[0]
; CHECK-NEXT:   mov.s  v1[2], v0[0]
; CHECK-NEXT:   mov.16b v0, v1
; CHECK-NEXT:   ret
;
entry:
  %0 = insertelement <4 x float> <float undef, float undef, float undef, float 0.000000e+00>, float %a, i32 0
  %1 = insertelement <4 x float> %0, float %a, i32 1
  %vecinit3 = insertelement <4 x float> %1, float %a, i32 2
  ret <4 x float> %vecinit3
}

define <4 x float> @test_insert_3_f32_undef(float %a) {
; CHECK-LABEL: test_insert_3_f32_undef:
; CHECK:       bb.0:
; CHECK-NEXT:   // kill
; CHECK-NEXT:   dup.4s  v0, v0[0]
; CHECK-NEXT:   ret
;
entry:
  %0 = insertelement <4 x float> <float undef, float undef, float undef, float undef>, float %a, i32 0
  %1 = insertelement <4 x float> %0, float %a, i32 1
  %vecinit3 = insertelement <4 x float> %1, float %a, i32 2
  ret <4 x float> %vecinit3
}

define <4 x float> @test_insert_2_f32_undef_zero(float %a) {
; CHECK-LABEL: test_insert_2_f32_undef_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    movi.2d v1, #0000000000000000
; CHECK-NEXT:    // kill
; CHECK-NEXT:    mov.s  v1[0], v0[0]
; CHECK-NEXT:    mov.s  v1[2], v0[0]
; CHECK-NEXT:    mov.16b v0, v1
; CHECK-NEXT:   ret
;
entry:
  %0 = insertelement <4 x float> <float undef, float 0.000000e+00, float undef, float 0.000000e+00>, float %a, i32 0
  %vecinit3 = insertelement <4 x float> %0, float %a, i32 2
  ret <4 x float> %vecinit3
}

define <4 x float> @test_insert_2_f32_var(float %a, <4 x float> %b) {
; CHECK-LABEL: test_insert_2_f32_var
; CHECK:       bb.0:
; CHECK-NEXT:    // kill
; CHECK-NEXT:   mov.s  v1[0], v0[0]
; CHECK-NEXT:   mov.s  v1[2], v0[0]
; CHECK-NEXT:   mov.16b v0, v1
; CHECK-NEXT:   ret
;
entry:
  %0 = insertelement <4 x float> %b, float %a, i32 0
  %vecinit3 = insertelement <4 x float> %0, float %a, i32 2
  ret <4 x float> %vecinit3
}
define <8 x i16> @test_insert_v8i16_i16_zero(<8 x i16> %a) {
; CHECK-LABEL: test_insert_v8i16_i16_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    mov.h   v0[5], wzr
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <8 x i16> %a, i16 0, i32 5
  ret <8 x i16> %vecinit5
}

; TODO: This should jsut be a mov.s v0[3], wzr
define <4 x half> @test_insert_v4f16_f16_zero(<4 x half> %a) {
; CHECK-LABEL: test_insert_v4f16_f16_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    adrp    x8, .LCPI7_0
; CHECK-NEXT:    kill
; CHECK-NEXT:    add x8, x8, :lo12:.LCPI7_0
; CHECK-NEXT:    ld1.h   { v0 }[0], [x8]
; CHECK-NEXT:    kill
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <4 x half> %a, half 0.000000e+00, i32 0
  ret <4 x half> %vecinit5
}

define <8 x half> @test_insert_v8f16_f16_zero(<8 x half> %a) {
; CHECK-LABEL: test_insert_v8f16_f16_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    adrp    x8, .LCPI8_0
; CHECK-NEXT:    add x8, x8, :lo12:.LCPI8_0
; CHECK-NEXT:    ld1.h   { v0 }[6], [x8]
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <8 x half> %a, half 0.000000e+00, i32 6
  ret <8 x half> %vecinit5
}

define <2 x float> @test_insert_v2f32_f32_zero(<2 x float> %a) {
; CHECK-LABEL: test_insert_v2f32_f32_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    // kill
; CHECK-NEXT:    fmov    s1, wzr
; CHECK-NEXT:    mov.s   v0[0], v1[0]
; CHECK-NEXT:    // kill
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <2 x float> %a, float 0.000000e+00, i32 0
  ret <2 x float> %vecinit5
}

define <4 x float> @test_insert_v4f32_f32_zero(<4 x float> %a) {
; CHECK-LABEL: test_insert_v4f32_f32_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    fmov    s1, wzr
; CHECK-NEXT:    mov.s   v0[3], v1[0]
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <4 x float> %a, float 0.000000e+00, i32 3
  ret <4 x float> %vecinit5
}

define <2 x double> @test_insert_v2f64_f64_zero(<2 x double> %a) {
; CHECK-LABEL: test_insert_v2f64_f64_zero:
; CHECK:       bb.0:
; CHECK-NEXT:    fmov    d1, xzr
; CHECK-NEXT:    mov.d   v0[1], v1[0]
; CHECK-NEXT:    ret

entry:
  %vecinit5 = insertelement <2 x double> %a, double 0.000000e+00, i32 1
  ret <2 x double> %vecinit5
}
