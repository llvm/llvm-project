; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=VBITS_EQ_128
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

; VBYTES represents the useful byte size of a vector register from the code
; generator's point of view. It is clamped to power-of-2 values because
; only power-of-2 vector lengths are considered legal, regardless of the
; user specified vector length.

; This test only tests the legal types for a given vector width, as mulh nodes
; do not get generated for non-legal types.

target triple = "aarch64-unknown-linux-gnu"

;
; SMULH
;

; Don't use SVE for 64-bit vectors.
; FIXME: The codegen for the >=256 bits case can be improved.
define <8 x i8> @smulh_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: smulh_v8i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smull v0.8h, v0.8b, v1.8b
; CHECK-NEXT:    ushr v1.8h, v0.8h, #8
; CHECK-NEXT:    umov w8, v1.h[0]
; CHECK-NEXT:    umov w9, v1.h[1]
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    umov w8, v1.h[2]
; CHECK-NEXT:    mov v0.b[1], w9
; CHECK-NEXT:    mov v0.b[2], w8
; CHECK-NEXT:    umov w8, v1.h[3]
; CHECK-NEXT:    mov v0.b[3], w8
; CHECK-NEXT:    umov w8, v1.h[4]
; CHECK-NEXT:    mov v0.b[4], w8
; CHECK-NEXT:    umov w8, v1.h[5]
; CHECK-NEXT:    mov v0.b[5], w8
; CHECK-NEXT:    umov w8, v1.h[6]
; CHECK-NEXT:    mov v0.b[6], w8
; CHECK-NEXT:    umov w8, v1.h[7]
; CHECK-NEXT:    mov v0.b[7], w8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %insert = insertelement <8 x i16> undef, i16 8, i64 0
  %splat = shufflevector <8 x i16> %insert, <8 x i16> undef, <8 x i32> zeroinitializer
  %1 = sext <8 x i8> %op1 to <8 x i16>
  %2 = sext <8 x i8> %op2 to <8 x i16>
  %mul = mul <8 x i16> %1, %2
  %shr = lshr <8 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <8 x i16> %shr to <8 x i8>
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @smulh_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: smulh_v16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smull2 v2.8h, v0.16b, v1.16b
; CHECK-NEXT:    smull v0.8h, v0.8b, v1.8b
; CHECK-NEXT:    uzp2 v0.16b, v0.16b, v2.16b
; CHECK-NEXT:    ret
  %1 = sext <16 x i8> %op1 to <16 x i16>
  %2 = sext <16 x i8> %op2 to <16 x i16>
  %mul = mul <16 x i16> %1, %2
  %shr = lshr <16 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <16 x i16> %shr to <16 x i8>
  ret <16 x i8> %res
}

define void @smulh_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; VBITS_GE_256-LABEL: smulh_v32i8:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.b, vl32
; VBITS_GE_256-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_256-NEXT:    smulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_256-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %1 = sext <32 x i8> %op1 to <32 x i16>
  %2 = sext <32 x i8> %op2 to <32 x i16>
  %mul = mul <32 x i16> %1, %2
  %shr = lshr <32 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <32 x i16> %shr to <32 x i8>
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @smulh_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; VBITS_GE_512-LABEL: smulh_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    smulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_512-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %insert = insertelement <64 x i16> undef, i16 8, i64 0
  %splat = shufflevector <64 x i16> %insert, <64 x i16> undef, <64 x i32> zeroinitializer
  %1 = sext <64 x i8> %op1 to <64 x i16>
  %2 = sext <64 x i8> %op2 to <64 x i16>
  %mul = mul <64 x i16> %1, %2
  %shr = lshr <64 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <64 x i16> %shr to <64 x i8>
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @smulh_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; VBITS_GE_1024-LABEL: smulh_v128i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl128
; VBITS_GE_1024-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    smulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_1024-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret

  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %1 = sext <128 x i8> %op1 to <128 x i16>
  %2 = sext <128 x i8> %op2 to <128 x i16>
  %mul = mul <128 x i16> %1, %2
  %shr = lshr <128 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <128 x i16> %shr to <128 x i8>
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @smulh_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; VBITS_GE_2048-LABEL: smulh_v256i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl256
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    smulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_2048-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %1 = sext <256 x i8> %op1 to <256 x i16>
  %2 = sext <256 x i8> %op2 to <256 x i16>
  %mul = mul <256 x i16> %1, %2
  %shr = lshr <256 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <256 x i16> %shr to <256 x i8>
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
; FIXME: The codegen for the >=256 bits case can be improved.
define <4 x i16> @smulh_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: smulh_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smull v0.4s, v0.4h, v1.4h
; CHECK-NEXT:    ushr v1.4s, v0.4s, #16
; CHECK-NEXT:    mov w8, v1.s[1]
; CHECK-NEXT:    mov w9, v1.s[2]
; CHECK-NEXT:    mov v0.16b, v1.16b
; CHECK-NEXT:    mov v0.h[1], w8
; CHECK-NEXT:    mov w8, v1.s[3]
; CHECK-NEXT:    mov v0.h[2], w9
; CHECK-NEXT:    mov v0.h[3], w8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %1 = sext <4 x i16> %op1 to <4 x i32>
  %2 = sext <4 x i16> %op2 to <4 x i32>
  %mul = mul <4 x i32> %1, %2
  %shr = lshr <4 x i32> %mul, <i32 16, i32 16, i32 16, i32 16>
  %res = trunc <4 x i32> %shr to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @smulh_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: smulh_v8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smull2 v2.4s, v0.8h, v1.8h
; CHECK-NEXT:    smull v0.4s, v0.4h, v1.4h
; CHECK-NEXT:    uzp2 v0.8h, v0.8h, v2.8h
; CHECK-NEXT:    ret
  %1 = sext <8 x i16> %op1 to <8 x i32>
  %2 = sext <8 x i16> %op2 to <8 x i32>
  %mul = mul <8 x i32> %1, %2
  %shr = lshr <8 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <8 x i32> %shr to <8 x i16>
  ret <8 x i16> %res
}

define void @smulh_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; VBITS_GE_256-LABEL: smulh_v16i16:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.h, vl16
; VBITS_GE_256-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_256-NEXT:    smulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_256-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %1 = sext <16 x i16> %op1 to <16 x i32>
  %2 = sext <16 x i16> %op2 to <16 x i32>
  %mul = mul <16 x i32> %1, %2
  %shr = lshr <16 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <16 x i32> %shr to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @smulh_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; VBITS_GE_512-LABEL: smulh_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    smulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %1 = sext <32 x i16> %op1 to <32 x i32>
  %2 = sext <32 x i16> %op2 to <32 x i32>
  %mul = mul <32 x i32> %1, %2
  %shr = lshr <32 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <32 x i32> %shr to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @smulh_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; VBITS_GE_1024-LABEL: smulh_v64i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    smulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %1 = sext <64 x i16> %op1 to <64 x i32>
  %2 = sext <64 x i16> %op2 to <64 x i32>
  %mul = mul <64 x i32> %1, %2
  %shr = lshr <64 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <64 x i32> %shr to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @smulh_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; VBITS_GE_2048-LABEL: smulh_v128i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    smulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %1 = sext <128 x i16> %op1 to <128 x i32>
  %2 = sext <128 x i16> %op2 to <128 x i32>
  %mul = mul <128 x i32> %1, %2
  %shr = lshr <128 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <128 x i32> %shr to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i32> @smulh_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: smulh_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sshll v0.2d, v0.2s, #0
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    sshll v1.2d, v1.2s, #0
; CHECK-NEXT:    mul z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    shrn v0.2s, v0.2d, #32
; CHECK-NEXT:    ret

; VBITS_EQ_128-LABEL: smulh_v2i32:
; VBITS_EQ_128:         sshll v0.2d, v0.2s, #0
; VBITS_EQ_128-NEXT:    ptrue p0.d, vl2
; VBITS_EQ_128-NEXT:    sshll v1.2d, v1.2s, #0
; VBITS_EQ_128-NEXT:    mul z0.d, p0/m, z0.d, z1.d
; VBITS_EQ_128-NEXT:    shrn v0.2s, v0.2d, #32
; VBITS_EQ_128-NEXT:    ret

  %1 = sext <2 x i32> %op1 to <2 x i64>
  %2 = sext <2 x i32> %op2 to <2 x i64>
  %mul = mul <2 x i64> %1, %2
  %shr = lshr <2 x i64> %mul, <i64 32, i64 32>
  %res = trunc <2 x i64> %shr to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @smulh_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: smulh_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    smull2 v2.2d, v0.4s, v1.4s
; CHECK-NEXT:    smull v0.2d, v0.2s, v1.2s
; CHECK-NEXT:    uzp2 v0.4s, v0.4s, v2.4s
; CHECK-NEXT:    ret
  %1 = sext <4 x i32> %op1 to <4 x i64>
  %2 = sext <4 x i32> %op2 to <4 x i64>
  %mul = mul <4 x i64> %1, %2
  %shr = lshr <4 x i64> %mul, <i64 32, i64 32, i64 32, i64 32>
  %res = trunc <4 x i64> %shr to <4 x i32>
  ret <4 x i32> %res
}

define void @smulh_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; VBITS_GE_256-LABEL: smulh_v8i32:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.s, vl8
; VBITS_GE_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_256-NEXT:    smulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_256-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %1 = sext <8 x i32> %op1 to <8 x i64>
  %2 = sext <8 x i32> %op2 to <8 x i64>
  %mul = mul <8 x i64> %1, %2
  %shr = lshr <8 x i64> %mul,  <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <8 x i64> %shr to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @smulh_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; VBITS_GE_512-LABEL: smulh_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    smulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %1 = sext <16 x i32> %op1 to <16 x i64>
  %2 = sext <16 x i32> %op2 to <16 x i64>
  %mul = mul <16 x i64> %1, %2
  %shr = lshr <16 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <16 x i64> %shr to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @smulh_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; VBITS_GE_1024-LABEL: smulh_v32i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    smulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %1 = sext <32 x i32> %op1 to <32 x i64>
  %2 = sext <32 x i32> %op2 to <32 x i64>
  %mul = mul <32 x i64> %1, %2
  %shr = lshr <32 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <32 x i64> %shr to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @smulh_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; VBITS_GE_2048-LABEL: smulh_v64i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    smulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %1 = sext <64 x i32> %op1 to <64 x i64>
  %2 = sext <64 x i32> %op2 to <64 x i64>
  %mul = mul <64 x i64> %1, %2
  %shr = lshr <64 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <64 x i64> %shr to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <1 x i64> @smulh_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: smulh_v1i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl1
; CHECK-NEXT:    // kill: def $d1 killed $d1 def $z1
; CHECK-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %insert = insertelement <1 x i128> undef, i128 64, i128 0
  %splat = shufflevector <1 x i128> %insert, <1 x i128> undef, <1 x i32> zeroinitializer
  %1 = sext <1 x i64> %op1 to <1 x i128>
  %2 = sext <1 x i64> %op2 to <1 x i128>
  %mul = mul <1 x i128> %1, %2
  %shr = lshr <1 x i128> %mul, %splat
  %res = trunc <1 x i128> %shr to <1 x i64>
  ret <1 x i64> %res
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i64> @smulh_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: smulh_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    // kill: def $q1 killed $q1 def $z1
; CHECK-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %1 = sext <2 x i64> %op1 to <2 x i128>
  %2 = sext <2 x i64> %op2 to <2 x i128>
  %mul = mul <2 x i128> %1, %2
  %shr = lshr <2 x i128> %mul, <i128 64, i128 64>
  %res = trunc <2 x i128> %shr to <2 x i64>
  ret <2 x i64> %res
}

define void @smulh_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: smulh_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %1 = sext <4 x i64> %op1 to <4 x i128>
  %2 = sext <4 x i64> %op2 to <4 x i128>
  %mul = mul <4 x i128> %1, %2
  %shr = lshr <4 x i128> %mul, <i128 64, i128 64, i128 64, i128 64>
  %res = trunc <4 x i128> %shr to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @smulh_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; VBITS_GE_512-LABEL: smulh_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %1 = sext <8 x i64> %op1 to <8 x i128>
  %2 = sext <8 x i64> %op2 to <8 x i128>
  %mul = mul <8 x i128> %1, %2
  %shr = lshr <8 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <8 x i128> %shr to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @smulh_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; VBITS_GE_1024-LABEL: smulh_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %1 = sext <16 x i64> %op1 to <16 x i128>
  %2 = sext <16 x i64> %op2 to <16 x i128>
  %mul = mul <16 x i128> %1, %2
  %shr = lshr <16 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <16 x i128> %shr to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @smulh_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; VBITS_GE_2048-LABEL: smulh_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    smulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %1 = sext <32 x i64> %op1 to <32 x i128>
  %2 = sext <32 x i64> %op2 to <32 x i128>
  %mul = mul <32 x i128> %1, %2
  %shr = lshr <32 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <32 x i128> %shr to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UMULH
;

; Don't use SVE for 64-bit vectors.
; FIXME: The codegen for the >=256 bits case can be improved.
define <8 x i8> @umulh_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: umulh_v8i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    umull v0.8h, v0.8b, v1.8b
; CHECK-NEXT:    ushr v1.8h, v0.8h, #8
; CHECK-NEXT:    umov w8, v1.h[0]
; CHECK-NEXT:    umov w9, v1.h[1]
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    umov w8, v1.h[2]
; CHECK-NEXT:    mov v0.b[1], w9
; CHECK-NEXT:    mov v0.b[2], w8
; CHECK-NEXT:    umov w8, v1.h[3]
; CHECK-NEXT:    mov v0.b[3], w8
; CHECK-NEXT:    umov w8, v1.h[4]
; CHECK-NEXT:    mov v0.b[4], w8
; CHECK-NEXT:    umov w8, v1.h[5]
; CHECK-NEXT:    mov v0.b[5], w8
; CHECK-NEXT:    umov w8, v1.h[6]
; CHECK-NEXT:    mov v0.b[6], w8
; CHECK-NEXT:    umov w8, v1.h[7]
; CHECK-NEXT:    mov v0.b[7], w8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %1 = zext <8 x i8> %op1 to <8 x i16>
  %2 = zext <8 x i8> %op2 to <8 x i16>
  %mul = mul <8 x i16> %1, %2
  %shr = lshr <8 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <8 x i16> %shr to <8 x i8>
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @umulh_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: umulh_v16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    umull2 v2.8h, v0.16b, v1.16b
; CHECK-NEXT:    umull v0.8h, v0.8b, v1.8b
; CHECK-NEXT:    uzp2 v0.16b, v0.16b, v2.16b
; CHECK-NEXT:    ret
  %1 = zext <16 x i8> %op1 to <16 x i16>
  %2 = zext <16 x i8> %op2 to <16 x i16>
  %mul = mul <16 x i16> %1, %2
  %shr = lshr <16 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <16 x i16> %shr to <16 x i8>
  ret <16 x i8> %res
}

define void @umulh_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; VBITS_GE_256-LABEL: umulh_v32i8:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.b, vl32
; VBITS_GE_256-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_256-NEXT:    umulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_256-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %1 = zext <32 x i8> %op1 to <32 x i16>
  %2 = zext <32 x i8> %op2 to <32 x i16>
  %mul = mul <32 x i16> %1, %2
  %shr = lshr <32 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <32 x i16> %shr to <32 x i8>
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @umulh_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; VBITS_GE_512-LABEL: umulh_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    umulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_512-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %1 = zext <64 x i8> %op1 to <64 x i16>
  %2 = zext <64 x i8> %op2 to <64 x i16>
  %mul = mul <64 x i16> %1, %2
  %shr = lshr <64 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <64 x i16> %shr to <64 x i8>
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @umulh_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; VBITS_GE_1024-LABEL: umulh_v128i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl128
; VBITS_GE_1024-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    umulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_1024-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret

  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %insert = insertelement <128 x i16> undef, i16 8, i64 0
  %splat = shufflevector <128 x i16> %insert, <128 x i16> undef, <128 x i32> zeroinitializer
  %1 = zext <128 x i8> %op1 to <128 x i16>
  %2 = zext <128 x i8> %op2 to <128 x i16>
  %mul = mul <128 x i16> %1, %2
  %shr = lshr <128 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <128 x i16> %shr to <128 x i8>
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @umulh_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; VBITS_GE_2048-LABEL: umulh_v256i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl256
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    umulh z0.b, p0/m, z0.b, z1.b
; VBITS_GE_2048-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %1 = zext <256 x i8> %op1 to <256 x i16>
  %2 = zext <256 x i8> %op2 to <256 x i16>
  %mul = mul <256 x i16> %1, %2
  %shr = lshr <256 x i16> %mul, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %res = trunc <256 x i16> %shr to <256 x i8>
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
; FIXME: The codegen for the >=256 bits case can be improved.
define <4 x i16> @umulh_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: umulh_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    umull v0.4s, v0.4h, v1.4h
; CHECK-NEXT:    ushr v1.4s, v0.4s, #16
; CHECK-NEXT:    mov w8, v1.s[1]
; CHECK-NEXT:    mov w9, v1.s[2]
; CHECK-NEXT:    mov v0.16b, v1.16b
; CHECK-NEXT:    mov v0.h[1], w8
; CHECK-NEXT:    mov w8, v1.s[3]
; CHECK-NEXT:    mov v0.h[2], w9
; CHECK-NEXT:    mov v0.h[3], w8
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $q0
; CHECK-NEXT:    ret
  %1 = zext <4 x i16> %op1 to <4 x i32>
  %2 = zext <4 x i16> %op2 to <4 x i32>
  %mul = mul <4 x i32> %1, %2
  %shr = lshr <4 x i32> %mul, <i32 16, i32 16, i32 16, i32 16>
  %res = trunc <4 x i32> %shr to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @umulh_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: umulh_v8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    umull2 v2.4s, v0.8h, v1.8h
; CHECK-NEXT:    umull v0.4s, v0.4h, v1.4h
; CHECK-NEXT:    uzp2 v0.8h, v0.8h, v2.8h
; CHECK-NEXT:    ret
  %1 = zext <8 x i16> %op1 to <8 x i32>
  %2 = zext <8 x i16> %op2 to <8 x i32>
  %mul = mul <8 x i32> %1, %2
  %shr = lshr <8 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <8 x i32> %shr to <8 x i16>
  ret <8 x i16> %res
}

define void @umulh_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; VBITS_GE_256-LABEL: umulh_v16i16:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.h, vl16
; VBITS_GE_256-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_256-NEXT:    umulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_256-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %1 = zext <16 x i16> %op1 to <16 x i32>
  %2 = zext <16 x i16> %op2 to <16 x i32>
  %mul = mul <16 x i32> %1, %2
  %shr = lshr <16 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <16 x i32> %shr to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @umulh_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; VBITS_GE_512-LABEL: umulh_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    umulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_512-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %1 = zext <32 x i16> %op1 to <32 x i32>
  %2 = zext <32 x i16> %op2 to <32 x i32>
  %mul = mul <32 x i32> %1, %2
  %shr = lshr <32 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <32 x i32> %shr to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @umulh_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; VBITS_GE_1024-LABEL: umulh_v64i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    umulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %1 = zext <64 x i16> %op1 to <64 x i32>
  %2 = zext <64 x i16> %op2 to <64 x i32>
  %mul = mul <64 x i32> %1, %2
  %shr = lshr <64 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <64 x i32> %shr to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @umulh_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; VBITS_GE_2048-LABEL: umulh_v128i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    umulh z0.h, p0/m, z0.h, z1.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %1 = zext <128 x i16> %op1 to <128 x i32>
  %2 = zext <128 x i16> %op2 to <128 x i32>
  %mul = mul <128 x i32> %1, %2
  %shr = lshr <128 x i32> %mul, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %res = trunc <128 x i32> %shr to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i32> @umulh_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: umulh_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ushll v1.2d, v1.2s, #0
; CHECK-NEXT:    mul z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    shrn v0.2s, v0.2d, #32
; CHECK-NEXT:    ret

; VBITS_EQ_128-LABEL: umulh_v2i32:
; VBITS_EQ_128:         ushll   v0.2d, v0.2s, #0
; VBITS_EQ_128-NEXT:    ptrue   p0.d, vl2
; VBITS_EQ_128-NEXT:    ushll   v1.2d, v1.2s, #0
; VBITS_EQ_128-NEXT:    mul     z0.d, p0/m, z0.d, z1.d
; VBITS_EQ_128-NEXT:    shrn    v0.2s, v0.2d, #32
; VBITS_EQ_128-NEXT:    ret

  %1 = zext <2 x i32> %op1 to <2 x i64>
  %2 = zext <2 x i32> %op2 to <2 x i64>
  %mul = mul <2 x i64> %1, %2
  %shr = lshr <2 x i64> %mul, <i64 32, i64 32>
  %res = trunc <2 x i64> %shr to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @umulh_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: umulh_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    umull2 v2.2d, v0.4s, v1.4s
; CHECK-NEXT:    umull v0.2d, v0.2s, v1.2s
; CHECK-NEXT:    uzp2 v0.4s, v0.4s, v2.4s
; CHECK-NEXT:    ret
  %1 = zext <4 x i32> %op1 to <4 x i64>
  %2 = zext <4 x i32> %op2 to <4 x i64>
  %mul = mul <4 x i64> %1, %2
  %shr = lshr <4 x i64> %mul, <i64 32, i64 32, i64 32, i64 32>
  %res = trunc <4 x i64> %shr to <4 x i32>
  ret <4 x i32> %res
}

define void @umulh_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; VBITS_GE_256-LABEL: umulh_v8i32:
; VBITS_GE_256:       // %bb.0:
; VBITS_GE_256-NEXT:    ptrue p0.s, vl8
; VBITS_GE_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_256-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_256-NEXT:    umulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_256-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_256-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %insert = insertelement <8 x i64> undef, i64 32, i64 0
  %splat = shufflevector <8 x i64> %insert, <8 x i64> undef, <8 x i32> zeroinitializer
  %1 = zext <8 x i32> %op1 to <8 x i64>
  %2 = zext <8 x i32> %op2 to <8 x i64>
  %mul = mul <8 x i64> %1, %2
  %shr = lshr <8 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <8 x i64> %shr to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @umulh_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; VBITS_GE_512-LABEL: umulh_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    umulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %1 = zext <16 x i32> %op1 to <16 x i64>
  %2 = zext <16 x i32> %op2 to <16 x i64>
  %mul = mul <16 x i64> %1, %2
  %shr = lshr <16 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <16 x i64> %shr to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @umulh_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; VBITS_GE_1024-LABEL: umulh_v32i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    umulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %1 = zext <32 x i32> %op1 to <32 x i64>
  %2 = zext <32 x i32> %op2 to <32 x i64>
  %mul = mul <32 x i64> %1, %2
  %shr = lshr <32 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <32 x i64> %shr to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @umulh_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; VBITS_GE_2048-LABEL: umulh_v64i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    umulh z0.s, p0/m, z0.s, z1.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %1 = zext <64 x i32> %op1 to <64 x i64>
  %2 = zext <64 x i32> %op2 to <64 x i64>
  %mul = mul <64 x i64> %1, %2
  %shr = lshr <64 x i64> %mul, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %res = trunc <64 x i64> %shr to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <1 x i64> @umulh_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: umulh_v1i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl1
; CHECK-NEXT:    // kill: def $d1 killed $d1 def $z1
; CHECK-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    // kill: def $d0 killed $d0 killed $z0
; CHECK-NEXT:    ret
  %1 = zext <1 x i64> %op1 to <1 x i128>
  %2 = zext <1 x i64> %op2 to <1 x i128>
  %mul = mul <1 x i128> %1, %2
  %shr = lshr <1 x i128> %mul, <i128 64>
  %res = trunc <1 x i128> %shr to <1 x i64>
  ret <1 x i64> %res
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i64> @umulh_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: umulh_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $q0 killed $q0 def $z0
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    // kill: def $q1 killed $q1 def $z1
; CHECK-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    // kill: def $q0 killed $q0 killed $z0
; CHECK-NEXT:    ret
  %1 = zext <2 x i64> %op1 to <2 x i128>
  %2 = zext <2 x i64> %op2 to <2 x i128>
  %mul = mul <2 x i128> %1, %2
  %shr = lshr <2 x i128> %mul, <i128 64, i128 64>
  %res = trunc <2 x i128> %shr to <2 x i64>
  ret <2 x i64> %res
}

define void @umulh_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: umulh_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %1 = zext <4 x i64> %op1 to <4 x i128>
  %2 = zext <4 x i64> %op2 to <4 x i128>
  %mul = mul <4 x i128> %1, %2
  %shr = lshr <4 x i128> %mul, <i128 64, i128 64, i128 64, i128 64>
  %res = trunc <4 x i128> %shr to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @umulh_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; VBITS_GE_512-LABEL: umulh_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %1 = zext <8 x i64> %op1 to <8 x i128>
  %2 = zext <8 x i64> %op2 to <8 x i128>
  %mul = mul <8 x i128> %1, %2
  %shr = lshr <8 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <8 x i128> %shr to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @umulh_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; VBITS_GE_1024-LABEL: umulh_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %1 = zext <16 x i64> %op1 to <16 x i128>
  %2 = zext <16 x i64> %op2 to <16 x i128>
  %mul = mul <16 x i128> %1, %2
  %shr = lshr <16 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <16 x i128> %shr to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @umulh_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; VBITS_GE_2048-LABEL: umulh_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    umulh z0.d, p0/m, z0.d, z1.d
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %1 = zext <32 x i64> %op1 to <32 x i128>
  %2 = zext <32 x i64> %op2 to <32 x i128>
  %mul = mul <32 x i128> %1, %2
  %shr = lshr <32 x i128> %mul, <i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64, i128 64>
  %res = trunc <32 x i128> %shr to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}
attributes #0 = { "target-features"="+sve" }
