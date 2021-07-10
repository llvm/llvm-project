; RUN: llc < %s -mtriple aarch64-apple-darwin -asm-verbose=false -disable-post-ra | FileCheck --check-prefixes=CHECK,NOFP16 %s
; RUN: llc < %s -mtriple aarch64-apple-darwin -asm-verbose=false -disable-post-ra -mattr=+v8.2a,+fullfp16 | FileCheck --check-prefixes=CHECK,FP16 %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

;============ v1f32

; WidenVecRes same
define <1 x float> @test_copysign_v1f32_v1f32(<1 x float> %a, <1 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v1f32_v1f32:
; CHECK-NEXT:    movi.2s v2, #128, lsl #24
; CHECK-NEXT:    bit.8b v0, v1, v2
; CHECK-NEXT:    ret
  %r = call <1 x float> @llvm.copysign.v1f32(<1 x float> %a, <1 x float> %b)
  ret <1 x float> %r
}

; WidenVecRes mismatched
define <1 x float> @test_copysign_v1f32_v1f64(<1 x float> %a, <1 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v1f32_v1f64:
; CHECK-NEXT:    fcvtn v1.2s, v1.2d
; CHECK-NEXT:    movi.2s v2, #128, lsl #24
; CHECK-NEXT:    bit.8b v0, v1, v2
; CHECK-NEXT:    ret
  %tmp0 = fptrunc <1 x double> %b to <1 x float>
  %r = call <1 x float> @llvm.copysign.v1f32(<1 x float> %a, <1 x float> %tmp0)
  ret <1 x float> %r
}

declare <1 x float> @llvm.copysign.v1f32(<1 x float> %a, <1 x float> %b) #0

;============ v1f64

; WidenVecOp #1
define <1 x double> @test_copysign_v1f64_v1f32(<1 x double> %a, <1 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v1f64_v1f32:
; CHECK-NEXT:    fcvtl v1.2d, v1.2s
; CHECK-NEXT:    movi.2d v2, #0000000000000000
; CHECK-NEXT:    fneg.2d v2, v2
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %tmp0 = fpext <1 x float> %b to <1 x double>
  %r = call <1 x double> @llvm.copysign.v1f64(<1 x double> %a, <1 x double> %tmp0)
  ret <1 x double> %r
}

define <1 x double> @test_copysign_v1f64_v1f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v1f64_v1f64:
; CHECK-NEXT:    movi.2d v2, #0000000000000000
; CHECK-NEXT:    fneg.2d v2, v2
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %r = call <1 x double> @llvm.copysign.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %r
}

declare <1 x double> @llvm.copysign.v1f64(<1 x double> %a, <1 x double> %b) #0

;============ v2f32

define <2 x float> @test_copysign_v2f32_v2f32(<2 x float> %a, <2 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v2f32_v2f32:
; CHECK-NEXT:    movi.2s v2, #128, lsl #24
; CHECK-NEXT:    bit.8b v0, v1, v2
; CHECK-NEXT:    ret
  %r = call <2 x float> @llvm.copysign.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

define <2 x float> @test_copysign_v2f32_v2f64(<2 x float> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v2f32_v2f64:
; CHECK-NEXT:    fcvtn v1.2s, v1.2d
; CHECK-NEXT:    movi.2s v2, #128, lsl #24
; CHECK-NEXT:    bit.8b v0, v1, v2
; CHECK-NEXT:    ret
  %tmp0 = fptrunc <2 x double> %b to <2 x float>
  %r = call <2 x float> @llvm.copysign.v2f32(<2 x float> %a, <2 x float> %tmp0)
  ret <2 x float> %r
}

declare <2 x float> @llvm.copysign.v2f32(<2 x float> %a, <2 x float> %b) #0

;============ v4f32

define <4 x float> @test_copysign_v4f32_v4f32(<4 x float> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v4f32_v4f32:
; CHECK-NEXT:    movi.4s v2, #128, lsl #24
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %r = call <4 x float> @llvm.copysign.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

; SplitVecOp #1
define <4 x float> @test_copysign_v4f32_v4f64(<4 x float> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v4f32_v4f64:
; CHECK-NEXT:    fcvtn v1.2s, v1.2d
; CHECK-NEXT:    fcvtn2 v1.4s, v2.2d
; CHECK-NEXT:    movi.4s v2, #128, lsl #24
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %tmp0 = fptrunc <4 x double> %b to <4 x float>
  %r = call <4 x float> @llvm.copysign.v4f32(<4 x float> %a, <4 x float> %tmp0)
  ret <4 x float> %r
}

declare <4 x float> @llvm.copysign.v4f32(<4 x float> %a, <4 x float> %b) #0

;============ v2f64

define <2 x double> @test_copysign_v2f64_v232(<2 x double> %a, <2 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v2f64_v232:
; CHECK-NEXT:    fcvtl v1.2d, v1.2s
; CHECK-NEXT:    movi.2d v2, #0000000000000000
; CHECK-NEXT:    fneg.2d v2, v2
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %tmp0 = fpext <2 x float> %b to <2 x double>
  %r = call <2 x double> @llvm.copysign.v2f64(<2 x double> %a, <2 x double> %tmp0)
  ret <2 x double> %r
}

define <2 x double> @test_copysign_v2f64_v2f64(<2 x double> %a, <2 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v2f64_v2f64:
; CHECK-NEXT:    movi.2d v2, #0000000000000000
; CHECK-NEXT:    fneg.2d v2, v2
; CHECK-NEXT:    bit.16b v0, v1, v2
; CHECK-NEXT:    ret
  %r = call <2 x double> @llvm.copysign.v2f64(<2 x double> %a, <2 x double> %b)
  ret <2 x double> %r
}

declare <2 x double> @llvm.copysign.v2f64(<2 x double> %a, <2 x double> %b) #0

;============ v4f64

; SplitVecRes mismatched
define <4 x double> @test_copysign_v4f64_v4f32(<4 x double> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v4f64_v4f32:
; CHECK-NEXT:    fcvtl v3.2d, v2.2s
; CHECK-NEXT:    fcvtl2 v2.2d, v2.4s
; CHECK-NEXT:    movi.2d v4, #0000000000000000
; CHECK-NEXT:    fneg.2d v4, v4
; CHECK-NEXT:    bit.16b v1, v2, v4
; CHECK-NEXT:    bit.16b v0, v3, v4
; CHECK-NEXT:    ret
  %tmp0 = fpext <4 x float> %b to <4 x double>
  %r = call <4 x double> @llvm.copysign.v4f64(<4 x double> %a, <4 x double> %tmp0)
  ret <4 x double> %r
}

; SplitVecRes same
define <4 x double> @test_copysign_v4f64_v4f64(<4 x double> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v4f64_v4f64:
; CHECK-NEXT:    movi.2d v4, #0000000000000000
; CHECK-NEXT:    fneg.2d v4, v4
; CHECK-NEXT:    bit.16b v0, v2, v4
; CHECK-NEXT:    bit.16b v1, v3, v4
; CHECK-NEXT:    ret
  %r = call <4 x double> @llvm.copysign.v4f64(<4 x double> %a, <4 x double> %b)
  ret <4 x double> %r
}

declare <4 x double> @llvm.copysign.v4f64(<4 x double> %a, <4 x double> %b) #0

;============ v4f16

define <4 x half> @test_copysign_v4f16_v4f16(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-LABEL: test_copysign_v4f16_v4f16:
; NOFP16-NEXT:    mov h2, v1[1]
; NOFP16-NEXT:    mov h3, v0[1]
; NOFP16-NEXT:    movi.4s v4, #128, lsl #24
; NOFP16-NEXT:    fcvt    s5, h1
; NOFP16-NEXT:    fcvt    s6, h0
; NOFP16-NEXT:    bit.16b v6, v5, v4
; NOFP16-NEXT:    mov h5, v1[2]
; NOFP16-NEXT:    fcvt    s2, h2
; NOFP16-NEXT:    fcvt    s3, h3
; NOFP16-NEXT:    bit.16b v3, v2, v4
; NOFP16-NEXT:    mov h2, v0[2]
; NOFP16-NEXT:    fcvt    s5, h5
; NOFP16-NEXT:    fcvt    s2, h2
; NOFP16-NEXT:    bit.16b v2, v5, v4
; NOFP16-NEXT:    mov h1, v1[3]
; NOFP16-NEXT:    mov h0, v0[3]
; NOFP16-NEXT:    fcvt    s1, h1
; NOFP16-NEXT:    fcvt    s5, h0
; NOFP16-NEXT:    fcvt    h0, s6
; NOFP16-NEXT:    bit.16b v5, v1, v4
; NOFP16-NEXT:    fcvt    h1, s3
; NOFP16-NEXT:    fcvt    h2, s2
; NOFP16-NEXT:    mov.h   v0[1], v1[0]
; NOFP16-NEXT:    mov.h   v0[2], v2[0]
; NOFP16-NEXT:    fcvt    h1, s5
; NOFP16-NEXT:    mov.h   v0[3], v1[0]
; NOFP16-NEXT:    ret

; FP16-NEXT:    movi.4h v2, #128, lsl #8
; FP16-NEXT:    bit.8b  v0, v1, v2
; FP16-NEXT:    ret
  %r = call <4 x half> @llvm.copysign.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %r
}

define <4 x half> @test_copysign_v4f16_v4f32(<4 x half> %a, <4 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v4f16_v4f32:
; NOFP16-NEXT:    fcvtn   v1.4h, v1.4s
; NOFP16-NEXT:    mov h2, v0[1]
; NOFP16-NEXT:    movi.4s v3, #128, lsl #24
; NOFP16-NEXT:    fcvt    s4, h0
; NOFP16-NEXT:    mov h5, v0[2]
; NOFP16-NEXT:    fcvt    s2, h2
; NOFP16-NEXT:    fcvt    s6, h1
; NOFP16-NEXT:    bit.16b v4, v6, v3
; NOFP16-NEXT:    mov h6, v1[1]
; NOFP16-NEXT:    fcvt    s5, h5
; NOFP16-NEXT:    fcvt    s6, h6
; NOFP16-NEXT:    bit.16b v2, v6, v3
; NOFP16-NEXT:    mov h6, v1[2]
; NOFP16-NEXT:    fcvt    s6, h6
; NOFP16-NEXT:    bit.16b v5, v6, v3
; NOFP16-NEXT:    mov h0, v0[3]
; NOFP16-NEXT:    fcvt    s6, h0
; NOFP16-NEXT:    mov h0, v1[3]
; NOFP16-NEXT:    fcvt    s1, h0
; NOFP16-NEXT:    fcvt    h0, s4
; NOFP16-NEXT:    bit.16b v6, v1, v3
; NOFP16-NEXT:    fcvt    h1, s2
; NOFP16-NEXT:    fcvt    h2, s5
; NOFP16-NEXT:    mov.h   v0[1], v1[0]
; NOFP16-NEXT:    mov.h   v0[2], v2[0]
; NOFP16-NEXT:    fcvt    h1, s6
; NOFP16-NEXT:    mov.h   v0[3], v1[0]
; NOFP16-NEXT:    ret

; FP16-NEXT:    fcvtn v1.4h, v1.4s
; FP16-NEXT:    movi.4h    v2, #128, lsl #8
; FP16-NEXT:    bit.8b v0, v1, v2
; FP16-NEXT:    ret
  %tmp0 = fptrunc <4 x float> %b to <4 x half>
  %r = call <4 x half> @llvm.copysign.v4f16(<4 x half> %a, <4 x half> %tmp0)
  ret <4 x half> %r
}

define <4 x half> @test_copysign_v4f16_v4f64(<4 x half> %a, <4 x double> %b) #0 {
; CHECK-LABEL: test_copysign_v4f16_v4f64:
; NOFP16-NEXT:    mov d3, v2[1]
; NOFP16-NEXT:    mov d4, v1[1]
; NOFP16-NEXT:    movi.4s v5, #128, lsl #24
; NOFP16-NEXT:    fcvt    s1, d1
; NOFP16-NEXT:    fcvt    s6, h0
; NOFP16-NEXT:    bit.16b v6, v1, v5
; NOFP16-NEXT:    mov h1, v0[1]
; NOFP16-NEXT:    fcvt    s2, d2
; NOFP16-NEXT:    fcvt    s4, d4
; NOFP16-NEXT:    fcvt    s1, h1
; NOFP16-NEXT:    bit.16b v1, v4, v5
; NOFP16-NEXT:    mov h4, v0[2]
; NOFP16-NEXT:    mov h0, v0[3]
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    fcvt    s3, d3
; NOFP16-NEXT:    fcvt    s7, h0
; NOFP16-NEXT:    fcvt    h0, s6
; NOFP16-NEXT:    bit.16b v4, v2, v5
; NOFP16-NEXT:    bit.16b v7, v3, v5
; NOFP16-NEXT:    fcvt    h1, s1
; NOFP16-NEXT:    fcvt    h2, s4
; NOFP16-NEXT:    mov.h   v0[1], v1[0]
; NOFP16-NEXT:    mov.h   v0[2], v2[0]
; NOFP16-NEXT:    fcvt    h1, s7
; NOFP16-NEXT:    mov.h   v0[3], v1[0]
; NOFP16-NEXT:    ret

; FP16-NEXT:    mov d3, v1[1]
; FP16-NEXT:    fcvt    h1, d1
; FP16-NEXT:    fcvt    h3, d3
; FP16-NEXT:    mov.h   v1[1], v3[0]
; FP16-NEXT:    fcvt    h3, d2
; FP16-NEXT:    mov d2, v2[1]
; FP16-NEXT:    fcvt    h2, d2
; FP16-NEXT:    mov.h   v1[2], v3[0]
; FP16-NEXT:    mov.h   v1[3], v2[0]
; FP16-NEXT:    movi.4h v2, #128, lsl #8
; FP16-NEXT:    bit.8b  v0, v1, v2
; FP16-NEXT:    ret
  %tmp0 = fptrunc <4 x double> %b to <4 x half>
  %r = call <4 x half> @llvm.copysign.v4f16(<4 x half> %a, <4 x half> %tmp0)
  ret <4 x half> %r
}

declare <4 x half> @llvm.copysign.v4f16(<4 x half> %a, <4 x half> %b) #0

;============ v8f16

define <8 x half> @test_copysign_v8f16_v8f16(<8 x half> %a, <8 x half> %b) #0 {
; CHECK-LABEL: test_copysign_v8f16_v8f16:
; NOFP16-NEXT:    mov h4, v1[1]
; NOFP16-NEXT:    mov h5, v0[1]
; NOFP16-NEXT:    movi.4s v2, #128, lsl #24
; NOFP16-NEXT:    fcvt    s6, h1
; NOFP16-NEXT:    fcvt    s3, h0
; NOFP16-NEXT:    mov h7, v1[2]
; NOFP16-NEXT:    mov h16, v0[2]
; NOFP16-NEXT:    mov h17, v1[3]
; NOFP16-NEXT:    mov h18, v0[3]
; NOFP16-NEXT:    bit.16b v3, v6, v2
; NOFP16-NEXT:    mov h6, v1[4]
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    fcvt    s5, h5
; NOFP16-NEXT:    bit.16b v5, v4, v2
; NOFP16-NEXT:    mov h4, v0[4]
; NOFP16-NEXT:    fcvt    s7, h7
; NOFP16-NEXT:    fcvt    s16, h16
; NOFP16-NEXT:    bit.16b v16, v7, v2
; NOFP16-NEXT:    mov h7, v1[5]
; NOFP16-NEXT:    fcvt    s17, h17
; NOFP16-NEXT:    fcvt    s18, h18
; NOFP16-NEXT:    bit.16b v18, v17, v2
; NOFP16-NEXT:    mov h17, v0[5]
; NOFP16-NEXT:    fcvt    s6, h6
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    bit.16b v4, v6, v2
; NOFP16-NEXT:    mov h6, v1[6]
; NOFP16-NEXT:    fcvt    s7, h7
; NOFP16-NEXT:    fcvt    s17, h17
; NOFP16-NEXT:    bit.16b v17, v7, v2
; NOFP16-NEXT:    mov h7, v0[6]
; NOFP16-NEXT:    fcvt    s6, h6
; NOFP16-NEXT:    fcvt    s7, h7
; NOFP16-NEXT:    bit.16b v7, v6, v2
; NOFP16-NEXT:    mov h1, v1[7]
; NOFP16-NEXT:    mov h0, v0[7]
; NOFP16-NEXT:    fcvt    s1, h1
; NOFP16-NEXT:    fcvt    s6, h0
; NOFP16-NEXT:    bit.16b v6, v1, v2
; NOFP16-NEXT:    fcvt    h0, s3
; NOFP16-NEXT:    fcvt    h1, s5
; NOFP16-NEXT:    mov.h   v0[1], v1[0]
; NOFP16-NEXT:    fcvt    h1, s16
; NOFP16-NEXT:    mov.h   v0[2], v1[0]
; NOFP16-NEXT:    fcvt    h1, s18
; NOFP16-NEXT:    fcvt    h2, s4
; NOFP16-NEXT:    fcvt    h3, s17
; NOFP16-NEXT:    fcvt    h4, s7
; NOFP16-NEXT:    mov.h   v0[3], v1[0]
; NOFP16-NEXT:    mov.h   v0[4], v2[0]
; NOFP16-NEXT:    mov.h   v0[5], v3[0]
; NOFP16-NEXT:    mov.h   v0[6], v4[0]
; NOFP16-NEXT:    fcvt    h1, s6
; NOFP16-NEXT:    mov.h   v0[7], v1[0]
; NOFP16-NEXT:    ret

; FP16-NEXT:    movi.8h v2, #128, lsl #8
; FP16-NEXT:    bit.16b  v0, v1, v2
; FP16-NEXT:    ret
  %r = call <8 x half> @llvm.copysign.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %r
}

define <8 x half> @test_copysign_v8f16_v8f32(<8 x half> %a, <8 x float> %b) #0 {
; CHECK-LABEL: test_copysign_v8f16_v8f32:
; NOFP16-NEXT:    fcvtn   v2.4h, v2.4s
; NOFP16-NEXT:    fcvtn   v4.4h, v1.4s
; NOFP16-NEXT:    mov h5, v0[1]
; NOFP16-NEXT:    movi.4s v1, #128, lsl #24
; NOFP16-NEXT:    fcvt    s3, h0
; NOFP16-NEXT:    mov h6, v0[2]
; NOFP16-NEXT:    mov h7, v0[3]
; NOFP16-NEXT:    mov h16, v0[4]
; NOFP16-NEXT:    mov h17, v0[5]
; NOFP16-NEXT:    fcvt    s5, h5
; NOFP16-NEXT:    fcvt    s18, h4
; NOFP16-NEXT:    fcvt    s16, h16
; NOFP16-NEXT:    bit.16b v3, v18, v1
; NOFP16-NEXT:    fcvt    s18, h2
; NOFP16-NEXT:    bit.16b v16, v18, v1
; NOFP16-NEXT:    mov h18, v4[1]
; NOFP16-NEXT:    fcvt    s6, h6
; NOFP16-NEXT:    fcvt    s18, h18
; NOFP16-NEXT:    bit.16b v5, v18, v1
; NOFP16-NEXT:    mov h18, v4[2]
; NOFP16-NEXT:    fcvt    s18, h18
; NOFP16-NEXT:    bit.16b v6, v18, v1
; NOFP16-NEXT:    mov h18, v0[6]
; NOFP16-NEXT:    fcvt    s7, h7
; NOFP16-NEXT:    mov h4, v4[3]
; NOFP16-NEXT:    fcvt    s17, h17
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    bit.16b v7, v4, v1
; NOFP16-NEXT:    mov h4, v2[1]
; NOFP16-NEXT:    fcvt    s18, h18
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    bit.16b v17, v4, v1
; NOFP16-NEXT:    mov h4, v2[2]
; NOFP16-NEXT:    fcvt    s4, h4
; NOFP16-NEXT:    bit.16b v18, v4, v1
; NOFP16-NEXT:    mov h0, v0[7]
; NOFP16-NEXT:    fcvt    s4, h0
; NOFP16-NEXT:    mov h0, v2[3]
; NOFP16-NEXT:    fcvt    s0, h0
; NOFP16-NEXT:    bit.16b v4, v0, v1
; NOFP16-NEXT:    fcvt    h0, s3
; NOFP16-NEXT:    fcvt    h1, s5
; NOFP16-NEXT:    mov.h   v0[1], v1[0]
; NOFP16-NEXT:    fcvt    h1, s16
; NOFP16-NEXT:    fcvt    h2, s6
; NOFP16-NEXT:    fcvt    h3, s7
; NOFP16-NEXT:    fcvt    h5, s17
; NOFP16-NEXT:    fcvt    h6, s18
; NOFP16-NEXT:    mov.h   v0[2], v2[0]
; NOFP16-NEXT:    mov.h   v0[3], v3[0]
; NOFP16-NEXT:    mov.h   v0[4], v1[0]
; NOFP16-NEXT:    mov.h   v0[5], v5[0]
; NOFP16-NEXT:    mov.h   v0[6], v6[0]
; NOFP16-NEXT:    fcvt    h1, s4
; NOFP16-NEXT:    mov.h   v0[7]
; NOFP16-NEXT:    ret

; FP16-NEXT:    fcvtn   v2.4h, v2.4s
; FP16-NEXT:    fcvtn   v1.4h, v1.4s
; FP16-NEXT:    mov.d   v1[1], v2[0]
; FP16-NEXT:    movi.8h v2, #128, lsl #8
; FP16-NEXT:    bit.16b v0, v1, v2
; FP16-NEXT:    ret
  %tmp0 = fptrunc <8 x float> %b to <8 x half>
  %r = call <8 x half> @llvm.copysign.v8f16(<8 x half> %a, <8 x half> %tmp0)
  ret <8 x half> %r
}

declare <8 x half> @llvm.copysign.v8f16(<8 x half> %a, <8 x half> %b) #0

attributes #0 = { nounwind }
