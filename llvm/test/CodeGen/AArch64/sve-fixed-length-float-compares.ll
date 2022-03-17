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
; NO_SVE-NOT: z{0-9}

;
; FCMP OEQ
;

; Don't use SVE for 64-bit vectors.
define <4 x i16> @fcmp_oeq_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <4 x half> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %sext
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @fcmp_oeq_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <8 x half> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}

define void @fcmp_oeq_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_oeq_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp oeq <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

define void @fcmp_oeq_v32f16(<32 x half>* %a, <32 x half>* %b, <32 x i16>* %c) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcmp_oeq_v32f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ld1h { z2.h }, p0/z, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z3.h }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    fcmeq p1.h, p0/z, z0.h, z2.h
; VBITS_EQ_256-NEXT:    fcmeq p2.h, p0/z, z1.h, z3.h
; VBITS_EQ_256-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    mov z1.h, p2/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    st1h { z0.h }, p0, [x2, x8, lsl #1]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcmp_oeq_v32f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p1.h, p0/z, z0.h, z1.h
; VBITS_GE_512-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x2]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %cmp = fcmp oeq <32 x half> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i16>
  store <32 x i16> %sext, <32 x i16>* %c
  ret void
}

define void @fcmp_oeq_v64f16(<64 x half>* %a, <64 x half>* %b, <64 x i16>* %c) #0 {
; VBITS_GE_1024-LABEL: fcmp_oeq_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p1.h, p0/z, z0.h, z1.h
; VBITS_GE_1024-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x2]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %cmp = fcmp oeq <64 x half> %op1, %op2
  %sext = sext <64 x i1> %cmp to <64 x i16>
  store <64 x i16> %sext, <64 x i16>* %c
  ret void
}

define void @fcmp_oeq_v128f16(<128 x half>* %a, <128 x half>* %b, <128 x i16>* %c) #0 {
; VBITS_GE_2048-LABEL: fcmp_oeq_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p1.h, p0/z, z0.h, z1.h
; VBITS_GE_2048-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x2]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %cmp = fcmp oeq <128 x half> %op1, %op2
  %sext = sext <128 x i1> %cmp to <128 x i16>
  store <128 x i16> %sext, <128 x i16>* %c
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @fcmp_oeq_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq v0.2s, v0.2s, v1.2s
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <2 x float> %op1, %op2
  %sext = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %sext
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @fcmp_oeq_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq v0.4s, v0.4s, v1.4s
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <4 x float> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}

define void @fcmp_oeq_v8f32(<8 x float>* %a, <8 x float>* %b, <8 x i32>* %c) #0 {
; CHECK-LABEL: fcmp_oeq_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w { z1.s }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; CHECK-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1w { z0.s }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %cmp = fcmp oeq <8 x float> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i32>
  store <8 x i32> %sext, <8 x i32>* %c
  ret void
}

define void @fcmp_oeq_v16f32(<16 x float>* %a, <16 x float>* %b, <16 x i32>* %c) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcmp_oeq_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ld1w { z2.s }, p0/z, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z3.s }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    fcmeq p1.s, p0/z, z0.s, z2.s
; VBITS_EQ_256-NEXT:    fcmeq p2.s, p0/z, z1.s, z3.s
; VBITS_EQ_256-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    mov z1.s, p2/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    st1w { z0.s }, p0, [x2, x8, lsl #2]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcmp_oeq_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_512-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x2]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %cmp = fcmp oeq <16 x float> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i32>
  store <16 x i32> %sext, <16 x i32>* %c
  ret void
}

define void @fcmp_oeq_v32f32(<32 x float>* %a, <32 x float>* %b, <32 x i32>* %c) #0 {
; VBITS_GE_1024-LABEL: fcmp_oeq_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_1024-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x2]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %cmp = fcmp oeq <32 x float> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i32>
  store <32 x i32> %sext, <32 x i32>* %c
  ret void
}

define void @fcmp_oeq_v64f32(<64 x float>* %a, <64 x float>* %b, <64 x i32>* %c) #0 {
; VBITS_GE_2048-LABEL: fcmp_oeq_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p1.s, p0/z, z0.s, z1.s
; VBITS_GE_2048-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x2]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %cmp = fcmp oeq <64 x float> %op1, %op2
  %sext = sext <64 x i1> %cmp to <64 x i32>
  store <64 x i32> %sext, <64 x i32>* %c
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcmp_oeq_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq d0, d0, d1
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <1 x double> %op1, %op2
  %sext = sext <1 x i1> %cmp to <1 x i64>
  ret <1 x i64> %sext
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @fcmp_oeq_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fcmp_oeq_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    fcmeq v0.2d, v0.2d, v1.2d
; CHECK-NEXT:    ret
  %cmp = fcmp oeq <2 x double> %op1, %op2
  %sext = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %sext
}

define void @fcmp_oeq_v4f64(<4 x double>* %a, <4 x double>* %b, <4 x i64>* %c) #0 {
; CHECK-LABEL: fcmp_oeq_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p1.d, p0/z, z0.d, z1.d
; CHECK-NEXT:    mov z0.d, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1d { z0.d }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %cmp = fcmp oeq <4 x double> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i64>
  store <4 x i64> %sext, <4 x i64>* %c
  ret void
}

define void @fcmp_oeq_v8f64(<8 x double>* %a, <8 x double>* %b, <8 x i64>* %c) #0 {
; Ensure sensible type legalisation
; VBITS_EQ_256-LABEL: fcmp_oeq_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z3.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    fcmeq p1.d, p0/z, z0.d, z2.d
; VBITS_EQ_256-NEXT:    fcmeq p2.d, p0/z, z1.d, z3.d
; VBITS_EQ_256-NEXT:    mov z0.d, p1/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    mov z1.d, p2/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x2, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: fcmp_oeq_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_512-NEXT:    mov z0.d, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x2]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %cmp = fcmp oeq <8 x double> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i64>
  store <8 x i64> %sext, <8 x i64>* %c
  ret void
}

define void @fcmp_oeq_v16f64(<16 x double>* %a, <16 x double>* %b, <16 x i64>* %c) #0 {
; VBITS_GE_1024-LABEL: fcmp_oeq_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_1024-NEXT:    mov z0.d, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x2]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %cmp = fcmp oeq <16 x double> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i64>
  store <16 x i64> %sext, <16 x i64>* %c
  ret void
}

define void @fcmp_oeq_v32f64(<32 x double>* %a, <32 x double>* %b, <32 x i64>* %c) #0 {
; VBITS_GE_2048-LABEL: fcmp_oeq_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p1.d, p0/z, z0.d, z1.d
; VBITS_GE_2048-NEXT:    mov z0.d, p1/z, #-1 // =0xffffffffffffffff
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x2]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %cmp = fcmp oeq <32 x double> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i64>
  store <32 x i64> %sext, <32 x i64>* %c
  ret void
}

;
; FCMP UEQ
;

define void @fcmp_ueq_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ueq_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmuo p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    fcmeq p2.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov p1.b, p2/m, p2.b
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ueq <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP ONE
;

define void @fcmp_one_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_one_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    fcmgt p2.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov p1.b, p2/m, p2.b
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp one <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP UNE
;

define void @fcmp_une_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_une_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmne p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp une <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP OGT
;

define void @fcmp_ogt_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ogt_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ogt <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP UGT
;

define void @fcmp_ugt_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ugt_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z1.h, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    eor z0.d, z0.d, z1.d
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ugt <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP OLT
;

define void @fcmp_olt_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_olt_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp olt <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP ULT
;

define void @fcmp_ult_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ult_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z1.h, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    eor z0.d, z0.d, z1.d
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ult <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP OGE
;

define void @fcmp_oge_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_oge_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp oge <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP UGE
;

define void @fcmp_uge_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_uge_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z1.h, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    eor z0.d, z0.d, z1.d
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp uge <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP OLE
;

define void @fcmp_ole_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ole_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ole <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP ULE
;

define void @fcmp_ule_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ule_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z1.h, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    eor z0.d, z0.d, z1.d
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ule <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP UNO
;

define void @fcmp_uno_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_uno_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmuo p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp uno <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP ORD
;

define void @fcmp_ord_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ord_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmuo p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z1.h, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    eor z0.d, z0.d, z1.d
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp ord <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP EQ
;

define void @fcmp_eq_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_eq_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast oeq <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP NE
;

define void @fcmp_ne_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ne_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmne p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast one <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP GT
;

define void @fcmp_gt_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_gt_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast ogt <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP LT
;

define void @fcmp_lt_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_lt_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmgt p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast olt <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP GE
;

define void @fcmp_ge_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_ge_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast oge <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

;
; FCMP LE
;

define void @fcmp_le_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: fcmp_le_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    fcmge p1.h, p0/z, z1.h, z0.h
; CHECK-NEXT:    mov z0.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    st1h { z0.h }, p0, [x2]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %cmp = fcmp fast ole <16 x half> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %c
  ret void
}

attributes #0 = { "target-features"="+sve" }
