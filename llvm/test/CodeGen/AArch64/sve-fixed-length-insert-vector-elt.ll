; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_256,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; insertelement
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @insertelement_v4f16(<4 x half> %op1) #0 {
; CHECK-LABEL: insertelement_v4f16:
; CHECK:         fmov h1, #5.00000000
; CHECK-NEXT:    mov v0.h[3], v1.h[0]
; CHECK-NEXT:    ret
    %r = insertelement <4 x half> %op1, half 5.0, i64 3
    ret <4 x half> %r
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @insertelement_v8f16(<8 x half> %op1) #0 {
; CHECK-LABEL: insertelement_v8f16:
; CHECK:         fmov h1, #5.00000000
; CHECK-NEXT:    mov v0.h[7], v1.h[0]
; CHECK-NEXT:    ret
    %r = insertelement <8 x half> %op1, half 5.0, i64 7
    ret <8 x half> %r
}

define <16 x half> @insertelement_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: insertelement_v16f16:
; VBITS_GE_256:         ptrue p0.h, vl16
; VBITS_GE_256-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov w9, #15
; VBITS_GE_256-NEXT:    mov z1.h, w9
; VBITS_GE_256-NEXT:    index z2.h, #0, #1
; VBITS_GE_256-NEXT:    ptrue p1.h
; VBITS_GE_256-NEXT:    cmpeq p1.h, p1/z, z2.h, z1.h
; VBITS_GE_256-NEXT:    fmov h1, #5.00000000
; VBITS_GE_256-NEXT:    mov z0.h, p1/m, h1
; VBITS_GE_256-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <16 x half>, <16 x half>* %a
    %r = insertelement <16 x half> %op1, half 5.0, i64 15
    ret <16 x half> %r
}

define <32 x half> @insertelement_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: insertelement_v32f16:
; VBITS_GE_512:         ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov w9, #31
; VBITS_GE_512-NEXT:    mov z1.h, w9
; VBITS_GE_512-NEXT:    index z2.h, #0, #1
; VBITS_GE_512-NEXT:    ptrue p1.h
; VBITS_GE_512-NEXT:    cmpeq p1.h, p1/z, z2.h, z1.h
; VBITS_GE_512-NEXT:    fmov h1, #5.00000000
; VBITS_GE_512-NEXT:    mov z0.h, p1/m, h1
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <32 x half>, <32 x half>* %a
    %r = insertelement <32 x half> %op1, half 5.0, i64 31
    ret <32 x half> %r
}

define <64 x half> @insertelement_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: insertelement_v64f16:
; VBITS_GE_1024:         ptrue   p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    mov     w9, #63
; VBITS_GE_1024-NEXT:    mov     z1.h, w9
; VBITS_GE_1024-NEXT:    index   z2.h, #0, #1
; VBITS_GE_1024-NEXT:    ptrue   p1.h
; VBITS_GE_1024-NEXT:    cmpeq   p1.h, p1/z, z2.h, z1.h
; VBITS_GE_1024-NEXT:    fmov    h1, #5.00000000
; VBITS_GE_1024-NEXT:    mov     z0.h, p1/m, h1
; VBITS_GE_1024-NEXT:    st1h    { z0.h }, p0, [x8]
; VBITS_GE_1024-NEXT:    ret
    %op1 = load <64 x half>, <64 x half>* %a
    %r = insertelement <64 x half> %op1, half 5.0, i64 63
    ret <64 x half> %r
}

define <128 x half> @insertelement_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: insertelement_v128f16:
; VBITS_GE_2048: ptrue   p0.h, vl128
; VBITS_GE_2048-NEXT: ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT: mov     w9, #127
; VBITS_GE_2048-NEXT: mov     z1.h, w9
; VBITS_GE_2048-NEXT: index   z2.h, #0, #1
; VBITS_GE_2048-NEXT: ptrue   p1.h
; VBITS_GE_2048-NEXT: cmpeq   p1.h, p1/z, z2.h, z1.h
; VBITS_GE_2048-NEXT: fmov    h1, #5.00000000
; VBITS_GE_2048-NEXT: mov     z0.h, p1/m, h1
; VBITS_GE_2048-NEXT: st1h    { z0.h }, p0, [x8]
; VBITS_GE_2048-NEXT: ret
    %op1 = load <128 x half>, <128 x half>* %a
    %r = insertelement <128 x half> %op1, half 5.0, i64 127
    ret <128 x half> %r
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @insertelement_v2f32(<2 x float> %op1) #0 {
; CHECK-LABEL: insertelement_v2f32:
; CHECK:         fmov s1, #5.00000000
; CHECK-NEXT:    mov v0.s[1], v1.s[0]
; CHECK-NEXT:    ret
    %r = insertelement <2 x float> %op1, float 5.0, i64 1
    ret <2 x float> %r
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @insertelement_v4f32(<4 x float> %op1) #0 {
; CHECK-LABEL: insertelement_v4f32:
; CHECK:         fmov s1, #5.00000000
; CHECK-NEXT:    mov v0.s[3], v1.s[0]
; CHECK-NEXT:    ret
    %r = insertelement <4 x float> %op1, float 5.0, i64 3
    ret <4 x float> %r
}

define <8 x float> @insertelement_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: insertelement_v8f32:
; VBITS_GE_256:         ptrue p0.s, vl8
; VBITS_GE_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov w9, #7
; VBITS_GE_256-NEXT:    mov z1.s, w9
; VBITS_GE_256-NEXT:    index z2.s, #0, #1
; VBITS_GE_256-NEXT:    ptrue p1.s
; VBITS_GE_256-NEXT:    cmpeq p1.s, p1/z, z2.s, z1.s
; VBITS_GE_256-NEXT:    fmov s1, #5.00000000
; VBITS_GE_256-NEXT:    mov z0.s, p1/m, s1
; VBITS_GE_256-NEXT:    st1w { z0.s }, p0, [x8]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <8 x float>, <8 x float>* %a
    %r = insertelement <8 x float> %op1, float 5.0, i64 7
    ret <8 x float> %r
}

define <16 x float> @insertelement_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: insertelement_v16f32:
; VBITS_GE_512:         ptrue   p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov     w9, #15
; VBITS_GE_512-NEXT:    mov     z1.s, w9
; VBITS_GE_512-NEXT:    index   z2.s, #0, #1
; VBITS_GE_512-NEXT:    ptrue   p1.s
; VBITS_GE_512-NEXT:    cmpeq   p1.s, p1/z, z2.s, z1.s
; VBITS_GE_512-NEXT:    fmov    s1, #5.00000000
; VBITS_GE_512-NEXT:    mov     z0.s, p1/m, s1
; VBITS_GE_512-NEXT:    st1w    { z0.s }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <16 x float>, <16 x float>* %a
    %r = insertelement <16 x float> %op1, float 5.0, i64 15
    ret <16 x float> %r
}

define <32 x float> @insertelement_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: insertelement_v32f32:
; VBITS_GE_1024:        ptrue   p0.s, vl32
; VBITS_GE_1024-NEXT:   ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:   mov     w9, #31
; VBITS_GE_1024-NEXT:   mov     z1.s, w9
; VBITS_GE_1024-NEXT:   index   z2.s, #0, #1
; VBITS_GE_1024-NEXT:   ptrue   p1.s
; VBITS_GE_1024-NEXT:   cmpeq   p1.s, p1/z, z2.s, z1.s
; VBITS_GE_1024-NEXT:   fmov    s1, #5.00000000
; VBITS_GE_1024-NEXT:   mov     z0.s, p1/m, s1
; VBITS_GE_1024-NEXT:   st1w    { z0.s }, p0, [x8]
; VBITS_GE_1024-NEXT:   ret
    %op1 = load <32 x float>, <32 x float>* %a
    %r = insertelement <32 x float> %op1, float 5.0, i64 31
    ret <32 x float> %r
}

define <64 x float> @insertelement_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: insertelement_v64f32:
; VBITS_GE_2048:        ptrue   p0.s, vl64
; VBITS_GE_2048-NEXT:   ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:   mov     w9, #63
; VBITS_GE_2048-NEXT:   mov     z1.s, w9
; VBITS_GE_2048-NEXT:   index   z2.s, #0, #1
; VBITS_GE_2048-NEXT:   ptrue   p1.s
; VBITS_GE_2048-NEXT:   cmpeq   p1.s, p1/z, z2.s, z1.s
; VBITS_GE_2048-NEXT:   fmov    s1, #5.00000000
; VBITS_GE_2048-NEXT:   mov     z0.s, p1/m, s1
; VBITS_GE_2048-NEXT:   st1w    { z0.s }, p0, [x8]
; VBITS_GE_2048-NEXT:   ret
    %op1 = load <64 x float>, <64 x float>* %a
    %r = insertelement <64 x float> %op1, float 5.0, i64 63
    ret <64 x float> %r
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @insertelement_v1f64(<1 x double> %op1) #0 {
; CHECK-LABEL: insertelement_v1f64:
; CHECK:         fmov d0, #5.00000000
; CHECK-NEXT:    ret
    %r = insertelement <1 x double> %op1, double 5.0, i64 0
    ret <1 x double> %r
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @insertelement_v2f64(<2 x double> %op1) #0 {
; CHECK-LABEL: insertelement_v2f64:
; CHECK:         fmov d1, #5.00000000
; CHECK-NEXT:    mov v0.d[1], v1.d[0]
; CHECK-NEXT:    ret
    %r = insertelement <2 x double> %op1, double 5.0, i64 1
    ret <2 x double> %r
}

define <4 x double> @insertelement_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: insertelement_v4f64:
; VBITS_GE_256:         ptrue p0.d, vl4
; VBITS_GE_256-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov w9, #3
; VBITS_GE_256-NEXT:    mov z1.d, x9
; VBITS_GE_256-NEXT:    index z2.d, #0, #1
; VBITS_GE_256-NEXT:    ptrue p1.d
; VBITS_GE_256-NEXT:    cmpeq p1.d, p1/z, z2.d, z1.d
; VBITS_GE_256-NEXT:    fmov d1, #5.00000000
; VBITS_GE_256-NEXT:    mov z0.d, p1/m, d1
; VBITS_GE_256-NEXT:    st1d { z0.d }, p0, [x8]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <4 x double>, <4 x double>* %a
    %r = insertelement <4 x double> %op1, double 5.0, i64 3
    ret <4 x double> %r
}

define <8 x double> @insertelement_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: insertelement_v8f64:
; VBITS_GE_512:         ptrue   p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov     w9, #7
; VBITS_GE_512-NEXT:    mov     z1.d, x9
; VBITS_GE_512-NEXT:    index   z2.d, #0, #1
; VBITS_GE_512-NEXT:    ptrue   p1.d
; VBITS_GE_512-NEXT:    cmpeq   p1.d, p1/z, z2.d, z1.d
; VBITS_GE_512-NEXT:    fmov    d1, #5.00000000
; VBITS_GE_512-NEXT:    mov     z0.d, p1/m, d1
; VBITS_GE_512-NEXT:    st1d    { z0.d }, p0, [x8]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <8 x double>, <8 x double>* %a
    %r = insertelement <8 x double> %op1, double 5.0, i64 7
    ret <8 x double> %r
}

define <16 x double> @insertelement_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: insertelement_v16f64:
; VBITS_GE_1024:         ptrue   p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    mov     w9, #15
; VBITS_GE_1024-NEXT:    mov     z1.d, x9
; VBITS_GE_1024-NEXT:    index   z2.d, #0, #1
; VBITS_GE_1024-NEXT:    ptrue   p1.d
; VBITS_GE_1024-NEXT:    cmpeq   p1.d, p1/z, z2.d, z1.d
; VBITS_GE_1024-NEXT:    fmov    d1, #5.00000000
; VBITS_GE_1024-NEXT:    mov     z0.d, p1/m, d1
; VBITS_GE_1024-NEXT:    st1d    { z0.d }, p0, [x8]
; VBITS_GE_1024-NEXT:    ret
    %op1 = load <16 x double>, <16 x double>* %a
    %r = insertelement <16 x double> %op1, double 5.0, i64 15
    ret <16 x double> %r
}

define <32 x double> @insertelement_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: insertelement_v32f64:
; VBITS_GE_2048:         ptrue   p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    mov     w9, #31
; VBITS_GE_2048-NEXT:    mov     z1.d, x9
; VBITS_GE_2048-NEXT:    index   z2.d, #0, #1
; VBITS_GE_2048-NEXT:    ptrue   p1.d
; VBITS_GE_2048-NEXT:    cmpeq   p1.d, p1/z, z2.d, z1.d
; VBITS_GE_2048-NEXT:    fmov    d1, #5.00000000
; VBITS_GE_2048-NEXT:    mov     z0.d, p1/m, d1
; VBITS_GE_2048-NEXT:    st1d    { z0.d }, p0, [x8]
; VBITS_GE_2048-NEXT:    ret
    %op1 = load <32 x double>, <32 x double>* %a
    %r = insertelement <32 x double> %op1, double 5.0, i64 31
    ret <32 x double> %r
}

attributes #0 = { "target-features"="+sve" }
