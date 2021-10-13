; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
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
; extractelement
;

; Don't use SVE for 64-bit vectors.
define half @extractelement_v4f16(<4 x half> %op1) #0 {
; CHECK-LABEL: extractelement_v4f16:
; CHECK:         mov h0, v0.h[3]
; CHECK-NEXT:    ret
    %r = extractelement <4 x half> %op1, i64 3
    ret half %r
}

; Don't use SVE for 128-bit vectors.
define half @extractelement_v8f16(<8 x half> %op1) #0 {
; CHECK-LABEL: extractelement_v8f16:
; CHECK:         mov h0, v0.h[7]
; CHECK-NEXT:    ret
    %r = extractelement <8 x half> %op1, i64 7
    ret half %r
}

define half @extractelement_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: extractelement_v16f16:
; VBITS_GE_256:         ptrue   p0.h, vl16
; VBITS_GE_256-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov z0.h, z0.h[15]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <16 x half>, <16 x half>* %a
    %r = extractelement <16 x half> %op1, i64 15
    ret half %r
}

define half @extractelement_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: extractelement_v32f16:
; VBITS_GE_512:         ptrue   p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov z0.h, z0.h[31]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <32 x half>, <32 x half>* %a
    %r = extractelement <32 x half> %op1, i64 31
    ret half %r
}

define half @extractelement_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: extractelement_v64f16:
; VBITS_GE_1024:         ptrue   p0.h, vl64
; VBITS_GE_1024-NEXT:    mov w8, #63
; VBITS_GE_1024-NEXT:    ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    whilels p0.h, xzr, x8
; VBITS_GE_1024-NEXT:    lastb   h0, p0, z0.h
; VBITS_GE_1024-NEXT:    ret
    %op1 = load <64 x half>, <64 x half>* %a
    %r = extractelement <64 x half> %op1, i64 63
    ret half %r
}

define half @extractelement_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: extractelement_v128f16:
; VBITS_GE_2048:      ptrue   p0.h, vl128
; VBITS_GE_2048-NEXT: mov w8, #127
; VBITS_GE_2048-NEXT: ld1h    { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT: whilels p0.h, xzr, x8
; VBITS_GE_2048-NEXT: lastb   h0, p0, z0.h
; VBITS_GE_2048-NEXT: ret
    %op1 = load <128 x half>, <128 x half>* %a
    %r = extractelement <128 x half> %op1, i64 127
    ret half %r
}

; Don't use SVE for 64-bit vectors.
define float @extractelement_v2f32(<2 x float> %op1) #0 {
; CHECK-LABEL: extractelement_v2f32:
; CHECK:         mov s0, v0.s[1]
; CHECK-NEXT:    ret
    %r = extractelement <2 x float> %op1, i64 1
    ret float %r
}

; Don't use SVE for 128-bit vectors.
define float @extractelement_v4f32(<4 x float> %op1) #0 {
; CHECK-LABEL: extractelement_v4f32:
; CHECK:         mov s0, v0.s[3]
; CHECK-NEXT:    ret
    %r = extractelement <4 x float> %op1, i64 3
    ret float %r
}

define float @extractelement_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: extractelement_v8f32:
; VBITS_GE_256:         ptrue   p0.s, vl8
; VBITS_GE_256-NEXT:    ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov z0.s, z0.s[7]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <8 x float>, <8 x float>* %a
    %r = extractelement <8 x float> %op1, i64 7
    ret float %r
}

define float @extractelement_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: extractelement_v16f32:
; VBITS_GE_512:         ptrue   p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov z0.s, z0.s[15]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <16 x float>, <16 x float>* %a
    %r = extractelement <16 x float> %op1, i64 15
    ret float %r
}

define float @extractelement_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: extractelement_v32f32:
; VBITS_GE_1024:        ptrue   p0.s, vl32
; VBITS_GE_1024-NEXT:   mov w8, #31
; VBITS_GE_1024-NEXT:   ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:   whilels p0.s, xzr, x8
; VBITS_GE_1024-NEXT:   lastb   s0, p0, z0.s
; VBITS_GE_1024-NEXT:   ret
    %op1 = load <32 x float>, <32 x float>* %a
    %r = extractelement <32 x float> %op1, i64 31
    ret float %r
}

define float @extractelement_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: extractelement_v64f32:
; VBITS_GE_2048:        ptrue   p0.s, vl64
; VBITS_GE_2048-NEXT:   mov w8, #63
; VBITS_GE_2048-NEXT:   ld1w    { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:   whilels p0.s, xzr, x8
; VBITS_GE_2048-NEXT:   lastb   s0, p0, z0.s
; VBITS_GE_2048-NEXT:   ret
    %op1 = load <64 x float>, <64 x float>* %a
    %r = extractelement <64 x float> %op1, i64 63
    ret float %r
}

; Don't use SVE for 64-bit vectors.
define double @extractelement_v1f64(<1 x double> %op1) #0 {
; CHECK-LABEL: extractelement_v1f64:
; CHECK:         ret
    %r = extractelement <1 x double> %op1, i64 0
    ret double %r
}

; Don't use SVE for 128-bit vectors.
define double @extractelement_v2f64(<2 x double> %op1) #0 {
; CHECK-LABEL: extractelement_v2f64:
; CHECK:         mov d0, v0.d[1]
; CHECK-NEXT:    ret
    %r = extractelement <2 x double> %op1, i64 1
    ret double %r
}

define double @extractelement_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: extractelement_v4f64:
; VBITS_GE_256:         ptrue   p0.d, vl4
; VBITS_GE_256-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_256-NEXT:    mov z0.d, z0.d[3]
; VBITS_GE_256-NEXT:    ret
    %op1 = load <4 x double>, <4 x double>* %a
    %r = extractelement <4 x double> %op1, i64 3
    ret double %r
}

define double @extractelement_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: extractelement_v8f64:
; VBITS_GE_512:         ptrue   p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    mov z0.d, z0.d[7]
; VBITS_GE_512-NEXT:    ret
    %op1 = load <8 x double>, <8 x double>* %a
    %r = extractelement <8 x double> %op1, i64 7
    ret double %r
}

define double @extractelement_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: extractelement_v16f64:
; VBITS_GE_1024:         ptrue   p0.d, vl16
; VBITS_GE_1024-NEXT:    mov w8, #15
; VBITS_GE_1024-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    whilels p0.d, xzr, x8
; VBITS_GE_1024-NEXT:    lastb   d0, p0, z0.d
; VBITS_GE_1024-NEXT:    ret
    %op1 = load <16 x double>, <16 x double>* %a
    %r = extractelement <16 x double> %op1, i64 15
    ret double %r
}

define double @extractelement_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: extractelement_v32f64:
; VBITS_GE_2048:         ptrue   p0.d, vl32
; VBITS_GE_2048-NEXT:    mov w8, #31
; VBITS_GE_2048-NEXT:    ld1d    { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    whilels p0.d, xzr, x8
; VBITS_GE_2048-NEXT:    lastb   d0, p0, z0.d
; VBITS_GE_2048-NEXT:    ret
    %op1 = load <32 x double>, <32 x double>* %a
    %r = extractelement <32 x double> %op1, i64 31
    ret double %r
}

attributes #0 = { "target-features"="+sve" }
