; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; FCVT H -> S
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v2f16_v2f32(<2 x half>* %a, <2 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v2f16_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s0, [x0]
; CHECK-NEXT:    fcvtl v0.4s, v0.4h
; CHECK-NEXT:    str d0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <2 x half>, <2 x half>* %a
  %res = fpext <2 x half> %op1 to <2 x float>
  store <2 x float> %res, <2 x float>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @fcvt_v4f16_v4f32(<4 x half>* %a, <4 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v4f16_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fcvtl v0.4s, v0.4h
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x half>, <4 x half>* %a
  %res = fpext <4 x half> %op1 to <4 x float>
  store <4 x float> %res, <4 x float>* %b
  ret void
}

define void @fcvt_v8f16_v8f32(<8 x half>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v8f16_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.s, p0/m, z0.h
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fpext <8 x half> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @fcvt_v16f16_v16f32(<16 x half>* %a, <16 x float>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: fcvt_v16f16_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1h { z0.s }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    fcvt z0.s, p0/m, z0.h
; VBITS_EQ_256-NEXT:    fcvt z1.s, p0/m, z1.h
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v16f16_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1h { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.s, p0/m, z0.h
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret

  %op1 = load <16 x half>, <16 x half>* %a
  %res = fpext <16 x half> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @fcvt_v32f16_v32f32(<32 x half>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v32f16_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1h { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.s, p0/m, z0.h
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fpext <32 x half> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @fcvt_v64f16_v64f32(<64 x half>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v64f16_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1h { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.s, p0/m, z0.h
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fpext <64 x half> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; FCVT H -> D
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v1f16_v1f64(<1 x half>* %a, <1 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v1f16_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr h0, [x0]
; CHECK-NEXT:    fcvt d0, h0
; CHECK-NEXT:    str d0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <1 x half>, <1 x half>* %a
  %res = fpext <1 x half> %op1 to <1 x double>
  store <1 x double> %res, <1 x double>* %b
  ret void
}

; v2f16 is not legal for NEON, so use SVE
define void @fcvt_v2f16_v2f64(<2 x half>* %a, <2 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v2f16_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    fcvt z0.d, p0/m, z0.h
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <2 x half>, <2 x half>* %a
  %res = fpext <2 x half> %op1 to <2 x double>
  store <2 x double> %res, <2 x double>* %b
  ret void
}

define void @fcvt_v4f16_v4f64(<4 x half>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v4f16_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.d, p0/m, z0.h
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x half>, <4 x half>* %a
  %res = fpext <4 x half> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @fcvt_v8f16_v8f64(<8 x half>* %a, <8 x double>* %b) #0 {
; VBITS_EQ_256-LABEL: fcvt_v8f16_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1h { z0.d }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    fcvt z0.d, p0/m, z0.h
; VBITS_EQ_256-NEXT:    fcvt z1.d, p0/m, z1.h
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v8f16_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1h { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.d, p0/m, z0.h
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret

  %op1 = load <8 x half>, <8 x half>* %a
  %res = fpext <8 x half> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @fcvt_v16f16_v16f64(<16 x half>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v16f16_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1h { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.d, p0/m, z0.h
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fpext <16 x half> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @fcvt_v32f16_v32f64(<32 x half>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v32f16_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.d, p0/m, z0.h
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fpext <32 x half> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; FCVT S -> D
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v1f32_v1f64(<1 x float>* %a, <1 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v1f32_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s0, [x0]
; CHECK-NEXT:    fcvtl v0.2d, v0.2s
; CHECK-NEXT:    str d0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <1 x float>, <1 x float>* %a
  %res = fpext <1 x float> %op1 to <1 x double>
  store <1 x double> %res, <1 x double>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @fcvt_v2f32_v2f64(<2 x float>* %a, <2 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v2f32_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fcvtl v0.2d, v0.2s
; CHECK-NEXT:    str q0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <2 x float>, <2 x float>* %a
  %res = fpext <2 x float> %op1 to <2 x double>
  store <2 x double> %res, <2 x double>* %b
  ret void
}

define void @fcvt_v4f32_v4f64(<4 x float>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v4f32_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.d, p0/m, z0.s
; CHECK-NEXT:    st1d { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x float>, <4 x float>* %a
  %res = fpext <4 x float> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @fcvt_v8f32_v8f64(<8 x float>* %a, <8 x double>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: fcvt_v8f32_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1w { z0.d }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    fcvt z0.d, p0/m, z0.s
; VBITS_EQ_256-NEXT:    fcvt z1.d, p0/m, z1.s
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v8f32_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.d, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fpext <8 x float> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @fcvt_v16f32_v16f64(<16 x float>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v16f32_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1w { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.d, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fpext <16 x float> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @fcvt_v32f32_v32f64(<32 x float>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v32f32_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.d, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fpext <32 x float> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; FCVT S -> H
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v2f32_v2f16(<2 x float>* %a, <2 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v2f32_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    str s0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <2 x float>, <2 x float>* %a
  %res = fptrunc <2 x float> %op1 to <2 x half>
  store <2 x half> %res, <2 x half>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @fcvt_v4f32_v4f16(<4 x float>* %a, <4 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v4f32_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    fcvtn v0.4h, v0.4s
; CHECK-NEXT:    str d0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x float>, <4 x float>* %a
  %res = fptrunc <4 x float> %op1 to <4 x half>
  store <4 x half> %res, <4 x half>* %b
  ret void
}

define void @fcvt_v8f32_v8f16(<8 x float>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v8f32_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.h, p0/m, z0.s
; CHECK-NEXT:    st1h { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptrunc <8 x float> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @fcvt_v16f32_v16f16(<16 x float>* %a, <16 x half>* %b) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcvt_v16f32_v16f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    fcvt z0.h, p0/m, z0.s
; VBITS_EQ_256-NEXT:    fcvt z1.h, p0/m, z1.s
; VBITS_EQ_256-NEXT:    st1h { z0.s }, p0, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z1.s }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v16f32_v16f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.h, p0/m, z0.s
; VBITS_GE_512-NEXT:    st1h { z0.s }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptrunc <16 x float> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @fcvt_v32f32_v32f16(<32 x float>* %a, <32 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v32f32_v32f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.h, p0/m, z0.s
; VBITS_GE_1024-NEXT:    st1h { z0.s }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptrunc <32 x float> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @fcvt_v64f32_v64f16(<64 x float>* %a, <64 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v64f32_v64f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.h, p0/m, z0.s
; VBITS_GE_2048-NEXT:    st1h { z0.s }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptrunc <64 x float> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; FCVT D -> H
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v1f64_v1f16(<1 x double>* %a, <1 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v1f64_v1f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fcvt h0, d0
; CHECK-NEXT:    str h0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <1 x double>, <1 x double>* %a
  %res = fptrunc <1 x double> %op1 to <1 x half>
  store <1 x half> %res, <1 x half>* %b
  ret void
}

; v2f16 is not legal for NEON, so use SVE
define void @fcvt_v2f64_v2f16(<2 x double>* %a, <2 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v2f64_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    fcvt z0.h, p0/m, z0.d
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    str s0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <2 x double>, <2 x double>* %a
  %res = fptrunc <2 x double> %op1 to <2 x half>
  store <2 x half> %res, <2 x half>* %b
  ret void
}

define void @fcvt_v4f64_v4f16(<4 x double>* %a, <4 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v4f64_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.h, p0/m, z0.d
; CHECK-NEXT:    st1h { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptrunc <4 x double> %op1 to <4 x half>
  store <4 x half> %res, <4 x half>* %b
  ret void
}

define void @fcvt_v8f64_v8f16(<8 x double>* %a, <8 x half>* %b) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcvt_v8f64_v8f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ptrue p0.d
; VBITS_EQ_256-NEXT:    fcvt z0.h, p0/m, z0.d
; VBITS_EQ_256-NEXT:    fcvt z1.h, p0/m, z1.d
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z1.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    mov v1.d[1], v0.d[0]
; VBITS_EQ_256-NEXT:    str q1, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v8f64_v8f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.h, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1h { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptrunc <8 x double> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @fcvt_v16f64_v16f16(<16 x double>* %a, <16 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v16f64_v16f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.h, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1h { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptrunc <16 x double> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @fcvt_v32f64_v32f16(<32 x double>* %a, <32 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v32f64_v32f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.h, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptrunc <32 x double> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; FCVT D -> S
;

; Don't use SVE for 64-bit vectors.
define void @fcvt_v1f64_v1f32(<1 x double> %op1, <1 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v1f64_v1f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    str s0, [x0]
; CHECK-NEXT:    ret
  %res = fptrunc <1 x double> %op1 to <1 x float>
  store <1 x float> %res, <1 x float>* %b
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @fcvt_v2f64_v2f32(<2 x double> %op1, <2 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v2f64_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcvtn v0.2s, v0.2d
; CHECK-NEXT:    str d0, [x0]
; CHECK-NEXT:    ret
  %res = fptrunc <2 x double> %op1 to <2 x float>
  store <2 x float> %res, <2 x float>* %b
  ret void
}

define void @fcvt_v4f64_v4f32(<4 x double>* %a, <4 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v4f64_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    fcvt z0.s, p0/m, z0.d
; CHECK-NEXT:    st1w { z0.d }, p0, [x1]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptrunc <4 x double> %op1 to <4 x float>
  store <4 x float> %res, <4 x float>* %b
  ret void
}

define void @fcvt_v8f64_v8f32(<8 x double>* %a, <8 x float>* %b) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcvt_v8f64_v8f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    fcvt z0.s, p0/m, z0.d
; VBITS_EQ_256-NEXT:    fcvt z1.s, p0/m, z1.d
; VBITS_EQ_256-NEXT:    st1w { z0.d }, p0, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.d }, p0, [x1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcvt_v8f64_v8f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    fcvt z0.s, p0/m, z0.d
; VBITS_GE_512-NEXT:    st1w { z0.d }, p0, [x1]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptrunc <8 x double> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @fcvt_v16f64_v16f32(<16 x double>* %a, <16 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: fcvt_v16f64_v16f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    fcvt z0.s, p0/m, z0.d
; VBITS_GE_1024-NEXT:    st1w { z0.d }, p0, [x1]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptrunc <16 x double> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @fcvt_v32f64_v32f32(<32 x double>* %a, <32 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: fcvt_v32f64_v32f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    fcvt z0.s, p0/m, z0.d
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [x1]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptrunc <32 x double> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
