; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK,VBITS_GE_256,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_256,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_2048,VBITS_GE_1024,VBITS_GE_512,VBITS_GE_256

; VBYTES represents the useful byte size of a vector register from the code
; generator's point of view. It is clamped to power-of-2 values because
; only power-of-2 vector lengths are considered legal, regardless of the
; user specified vector length.

; This test only tests the legal types for a given vector width, as mulh nodes
; do not get generated for non-legal types.

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; SMULH
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @smulh_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: smulh_v8i8:
; CHECK: smull v0.8h, v0.8b, v1.8b
; CHECK: ushr v1.8h, v0.8h, #8
; CHECK: umov w8, v1.h[0]
; CHECK: fmov s0, w8
; CHECK: umov w8, v1.h[1]
; CHECK: mov v0.b[1], w8
; CHECK: umov w8, v1.h[2]
; CHECK: mov v0.b[2], w8
; CHECK: umov w8, v1.h[3]
; CHECK: mov v0.b[3], w8
; CHECK: ret
  %insert = insertelement <8 x i16> undef, i16 8, i64 0
  %splat = shufflevector <8 x i16> %insert, <8 x i16> undef, <8 x i32> zeroinitializer
  %1 = sext <8 x i8> %op1 to <8 x i16>
  %2 = sext <8 x i8> %op2 to <8 x i16>
  %mul = mul <8 x i16> %1, %2
  %shr = lshr <8 x i16> %mul, %splat
  %res = trunc <8 x i16> %shr to <8 x i8>
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @smulh_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: smulh_v16i8:
; CHECK: smull2 v2.8h, v0.16b, v1.16b
; CHECK: smull v0.8h, v0.8b, v1.8b
; CHECK: uzp2 v0.16b, v0.16b, v2.16b
; CHECK: ret
  %insert = insertelement <16 x i16> undef, i16 8, i64 0
  %splat = shufflevector <16 x i16> %insert, <16 x i16> undef, <16 x i32> zeroinitializer
  %1 = sext <16 x i8> %op1 to <16 x i16>
  %2 = sext <16 x i8> %op2 to <16 x i16>
  %mul = mul <16 x i16> %1, %2
  %shr = lshr <16 x i16> %mul, %splat
  %res = trunc <16 x i16> %shr to <16 x i8>
  ret <16 x i8> %res
}

define void @smulh_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: smulh_v32i8:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; VBITS_EQ_256-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256: smulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,32)]]
; VBITS_GE_512-DAG: ld1sb { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1sb { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_512: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %insert = insertelement <32 x i16> undef, i16 8, i64 0
  %splat = shufflevector <32 x i16> %insert, <32 x i16> undef, <32 x i32> zeroinitializer
  %1 = sext <32 x i8> %op1 to <32 x i16>
  %2 = sext <32 x i8> %op2 to <32 x i16>
  %mul = mul <32 x i16> %1, %2
  %shr = lshr <32 x i16> %mul, %splat
  %res = trunc <32 x i16> %shr to <32 x i8>
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @smulh_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: smulh_v64i8:
; VBITS_EQ_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_512: smulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_512: ret

; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,64)]]
; VBITS_GE_1024-DAG: ld1sb { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1sb { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_1024: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %insert = insertelement <64 x i16> undef, i16 8, i64 0
  %splat = shufflevector <64 x i16> %insert, <64 x i16> undef, <64 x i32> zeroinitializer
  %1 = sext <64 x i8> %op1 to <64 x i16>
  %2 = sext <64 x i8> %op2 to <64 x i16>
  %mul = mul <64 x i16> %1, %2
  %shr = lshr <64 x i16> %mul, %splat
  %res = trunc <64 x i16> %shr to <64 x i8>
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @smulh_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: smulh_v128i8:
; VBITS_EQ_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_1024: smulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,128)]]
; VBITS_GE_2048-DAG: ld1sb { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1sb { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_2048: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %insert = insertelement <128 x i16> undef, i16 8, i64 0
  %splat = shufflevector <128 x i16> %insert, <128 x i16> undef, <128 x i32> zeroinitializer
  %1 = sext <128 x i8> %op1 to <128 x i16>
  %2 = sext <128 x i8> %op2 to <128 x i16>
  %mul = mul <128 x i16> %1, %2
  %shr = lshr <128 x i16> %mul, %splat
  %res = trunc <128 x i16> %shr to <128 x i8>
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @smulh_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: smulh_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048: smulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %insert = insertelement <256 x i16> undef, i16 8, i64 0
  %splat = shufflevector <256 x i16> %insert, <256 x i16> undef, <256 x i32> zeroinitializer
  %1 = sext <256 x i8> %op1 to <256 x i16>
  %2 = sext <256 x i8> %op2 to <256 x i16>
  %mul = mul <256 x i16> %1, %2
  %shr = lshr <256 x i16> %mul, %splat
  %res = trunc <256 x i16> %shr to <256 x i8>
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @smulh_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: smulh_v4i16:
; CHECK: smull v0.4s, v0.4h, v1.4h
; CHECK: ushr v0.4s, v0.4s, #16
; CHECK: mov w8, v0.s[1]
; CHECK: mov w9, v0.s[2]
; CHECK: mov w10, v0.s[3]
; CHECK: mov v0.h[1], w8
; CHECK: mov v0.h[2], w9
; CHECK: mov v0.h[3], w10
; CHECK: ret
  %insert = insertelement <4 x i32> undef, i32 16, i64 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %1 = sext <4 x i16> %op1 to <4 x i32>
  %2 = sext <4 x i16> %op2 to <4 x i32>
  %mul = mul <4 x i32> %1, %2
  %shr = lshr <4 x i32> %mul, %splat
  %res = trunc <4 x i32> %shr to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @smulh_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: smulh_v8i16:
; CHECK: smull2 v2.4s, v0.8h, v1.8h
; CHECK: smull v0.4s, v0.4h, v1.4h
; CHECK: uzp2 v0.8h, v0.8h, v2.8h
; CHECK: ret
  %insert = insertelement <8 x i32> undef, i32 16, i64 0
  %splat = shufflevector <8 x i32> %insert, <8 x i32> undef, <8 x i32> zeroinitializer
  %1 = sext <8 x i16> %op1 to <8 x i32>
  %2 = sext <8 x i16> %op2 to <8 x i32>
  %mul = mul <8 x i32> %1, %2
  %shr = lshr <8 x i32> %mul, %splat
  %res = trunc <8 x i32> %shr to <8 x i16>
  ret <8 x i16> %res
}

define void @smulh_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: smulh_v16i16:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,16)]]
; VBITS_EQ_256-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256: smulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_256: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,16)]]
; VBITS_GE_512-DAG: ld1sh { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1sh { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_512: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %insert = insertelement <16 x i32> undef, i32 16, i64 0
  %splat = shufflevector <16 x i32> %insert, <16 x i32> undef, <16 x i32> zeroinitializer
  %1 = sext <16 x i16> %op1 to <16 x i32>
  %2 = sext <16 x i16> %op2 to <16 x i32>
  %mul = mul <16 x i32> %1, %2
  %shr = lshr <16 x i32> %mul, %splat
  %res = trunc <16 x i32> %shr to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @smulh_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: smulh_v32i16:
; VBITS_EQ_512: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,32)]]
; VBITS_EQ_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_512: smulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_512: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_512: ret

; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,32)]]
; VBITS_GE_1024-DAG: ld1sh { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1sh { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_1024: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %insert = insertelement <32 x i32> undef, i32 16, i64 0
  %splat = shufflevector <32 x i32> %insert, <32 x i32> undef, <32 x i32> zeroinitializer
  %1 = sext <32 x i16> %op1 to <32 x i32>
  %2 = sext <32 x i16> %op2 to <32 x i32>
  %mul = mul <32 x i32> %1, %2
  %shr = lshr <32 x i32> %mul, %splat
  %res = trunc <32 x i32> %shr to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @smulh_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: smulh_v64i16:
; VBITS_EQ_1024: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,64)]]
; VBITS_EQ_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_1024: smulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_1024: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,64)]]
; VBITS_GE_2048-DAG: ld1sh { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1sh { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_2048: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %insert = insertelement <64 x i32> undef, i32 16, i64 0
  %splat = shufflevector <64 x i32> %insert, <64 x i32> undef, <64 x i32> zeroinitializer
  %1 = sext <64 x i16> %op1 to <64 x i32>
  %2 = sext <64 x i16> %op2 to <64 x i32>
  %mul = mul <64 x i32> %1, %2
  %shr = lshr <64 x i32> %mul, %splat
  %res = trunc <64 x i32> %shr to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @smulh_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: smulh_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,128)]]
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048: smulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %insert = insertelement <128 x i32> undef, i32 16, i64 0
  %splat = shufflevector <128 x i32> %insert, <128 x i32> undef, <128 x i32> zeroinitializer
  %1 = sext <128 x i16> %op1 to <128 x i32>
  %2 = sext <128 x i16> %op2 to <128 x i32>
  %mul = mul <128 x i32> %1, %2
  %shr = lshr <128 x i32> %mul, %splat
  %res = trunc <128 x i32> %shr to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i32> @smulh_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: smulh_v2i32:
; CHECK: sshll v0.2d, v0.2s, #0
; CHECK: sshll v1.2d, v1.2s, #0
; CHECK: ptrue p0.d, vl2
; CHECK: mul z0.d, p0/m, z0.d, z1.d
; CHECK: shrn v0.2s, v0.2d, #32
; CHECK: ret
  %insert = insertelement <2 x i64> undef, i64 32, i64 0
  %splat = shufflevector <2 x i64> %insert, <2 x i64> undef, <2 x i32> zeroinitializer
  %1 = sext <2 x i32> %op1 to <2 x i64>
  %2 = sext <2 x i32> %op2 to <2 x i64>
  %mul = mul <2 x i64> %1, %2
  %shr = lshr <2 x i64> %mul, %splat
  %res = trunc <2 x i64> %shr to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @smulh_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: smulh_v4i32:
; CHECK: smull2 v2.2d, v0.4s, v1.4s
; CHECK: smull v0.2d, v0.2s, v1.2s
; CHECK: uzp2 v0.4s, v0.4s, v2.4s
; CHECK: ret
  %insert = insertelement <4 x i64> undef, i64 32, i64 0
  %splat = shufflevector <4 x i64> %insert, <4 x i64> undef, <4 x i32> zeroinitializer
  %1 = sext <4 x i32> %op1 to <4 x i64>
  %2 = sext <4 x i32> %op2 to <4 x i64>
  %mul = mul <4 x i64> %1, %2
  %shr = lshr <4 x i64> %mul, %splat
  %res = trunc <4 x i64> %shr to <4 x i32>
  ret <4 x i32> %res
}

define void @smulh_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: smulh_v8i32:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,8)]]
; VBITS_EQ_256-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256: smulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_256: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,8)]]
; VBITS_GE_512-DAG: ld1sw { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1sw { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512: lsr [[RES]].d, [[PG]]/m, [[RES]].d, #32
; VBITS_GE_512: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %insert = insertelement <8 x i64> undef, i64 32, i64 0
  %splat = shufflevector <8 x i64> %insert, <8 x i64> undef, <8 x i32> zeroinitializer
  %1 = sext <8 x i32> %op1 to <8 x i64>
  %2 = sext <8 x i32> %op2 to <8 x i64>
  %mul = mul <8 x i64> %1, %2
  %shr = lshr <8 x i64> %mul, %splat
  %res = trunc <8 x i64> %shr to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @smulh_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: smulh_v16i32:
; VBITS_EQ_512: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,16)]]
; VBITS_EQ_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_512: smulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_512: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_512: ret

; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,16)]]
; VBITS_GE_1024-DAG: ld1sw { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1sw { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %insert = insertelement <16 x i64> undef, i64 32, i64 0
  %splat = shufflevector <16 x i64> %insert, <16 x i64> undef, <16 x i32> zeroinitializer
  %1 = sext <16 x i32> %op1 to <16 x i64>
  %2 = sext <16 x i32> %op2 to <16 x i64>
  %mul = mul <16 x i64> %1, %2
  %shr = lshr <16 x i64> %mul, %splat
  %res = trunc <16 x i64> %shr to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @smulh_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: smulh_v32i32:
; VBITS_EQ_1024: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,32)]]
; VBITS_EQ_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_1024: smulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_1024: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,32)]]
; VBITS_GE_2048-DAG: ld1sw { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1sw { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048: lsr [[RES]].d, [[PG]]/m, [[RES]].d, #32
; VBITS_GE_2048: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %insert = insertelement <32 x i64> undef, i64 32, i64 0
  %splat = shufflevector <32 x i64> %insert, <32 x i64> undef, <32 x i32> zeroinitializer
  %1 = sext <32 x i32> %op1 to <32 x i64>
  %2 = sext <32 x i32> %op2 to <32 x i64>
  %mul = mul <32 x i64> %1, %2
  %shr = lshr <32 x i64> %mul, %splat
  %res = trunc <32 x i64> %shr to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @smulh_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: smulh_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,64)]]
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048: smulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %insert = insertelement <64 x i64> undef, i64 32, i64 0
  %splat = shufflevector <64 x i64> %insert, <64 x i64> undef, <64 x i32> zeroinitializer
  %1 = sext <64 x i32> %op1 to <64 x i64>
  %2 = sext <64 x i32> %op2 to <64 x i64>
  %mul = mul <64 x i64> %1, %2
  %shr = lshr <64 x i64> %mul, %splat
  %res = trunc <64 x i64> %shr to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <1 x i64> @smulh_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: smulh_v1i64:
; CHECK: ptrue p0.d, vl1
; CHECK: smulh z0.d, p0/m, z0.d, z1.d
; CHECK: ret
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
; CHECK: ptrue p0.d, vl2
; CHECK: smulh z0.d, p0/m, z0.d, z1.d
; CHECK: ret
  %insert = insertelement <2 x i128> undef, i128 64, i128 0
  %splat = shufflevector <2 x i128> %insert, <2 x i128> undef, <2 x i32> zeroinitializer
  %1 = sext <2 x i64> %op1 to <2 x i128>
  %2 = sext <2 x i64> %op2 to <2 x i128>
  %mul = mul <2 x i128> %1, %2
  %shr = lshr <2 x i128> %mul, %splat
  %res = trunc <2 x i128> %shr to <2 x i64>
  ret <2 x i64> %res
}

define void @smulh_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: smulh_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,4)]]
; VBITS_GE_256-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_256: smulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_256: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_256: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %insert = insertelement <4 x i128> undef, i128 64, i128 0
  %splat = shufflevector <4 x i128> %insert, <4 x i128> undef, <4 x i32> zeroinitializer
  %1 = sext <4 x i64> %op1 to <4 x i128>
  %2 = sext <4 x i64> %op2 to <4 x i128>
  %mul = mul <4 x i128> %1, %2
  %shr = lshr <4 x i128> %mul, %splat
  %res = trunc <4 x i128> %shr to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @smulh_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: smulh_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,8)]]
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512: smulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %insert = insertelement <8 x i128> undef, i128 64, i128 0
  %splat = shufflevector <8 x i128> %insert, <8 x i128> undef, <8 x i32> zeroinitializer
  %1 = sext <8 x i64> %op1 to <8 x i128>
  %2 = sext <8 x i64> %op2 to <8 x i128>
  %mul = mul <8 x i128> %1, %2
  %shr = lshr <8 x i128> %mul, %splat
  %res = trunc <8 x i128> %shr to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @smulh_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: smulh_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,16)]]
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024: smulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %insert = insertelement <16 x i128> undef, i128 64, i128 0
  %splat = shufflevector <16 x i128> %insert, <16 x i128> undef, <16 x i32> zeroinitializer
  %1 = sext <16 x i64> %op1 to <16 x i128>
  %2 = sext <16 x i64> %op2 to <16 x i128>
  %mul = mul <16 x i128> %1, %2
  %shr = lshr <16 x i128> %mul, %splat
  %res = trunc <16 x i128> %shr to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @smulh_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: smulh_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,32)]]
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048: smulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %insert = insertelement <32 x i128> undef, i128 64, i128 0
  %splat = shufflevector <32 x i128> %insert, <32 x i128> undef, <32 x i32> zeroinitializer
  %1 = sext <32 x i64> %op1 to <32 x i128>
  %2 = sext <32 x i64> %op2 to <32 x i128>
  %mul = mul <32 x i128> %1, %2
  %shr = lshr <32 x i128> %mul, %splat
  %res = trunc <32 x i128> %shr to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UMULH
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @umulh_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: umulh_v8i8:
; CHECK: umull v0.8h, v0.8b, v1.8b
; CHECK: ushr v1.8h, v0.8h, #8
; CHECK: umov w8, v1.h[0]
; CHECK: fmov s0, w8
; CHECK: umov w8, v1.h[1]
; CHECK: mov v0.b[1], w8
; CHECK: umov w8, v1.h[2]
; CHECK: mov v0.b[2], w8
; CHECK: umov w8, v1.h[3]
; CHECK: mov v0.b[3], w8
; CHECK: ret
  %insert = insertelement <8 x i16> undef, i16 8, i64 0
  %splat = shufflevector <8 x i16> %insert, <8 x i16> undef, <8 x i32> zeroinitializer
  %1 = zext <8 x i8> %op1 to <8 x i16>
  %2 = zext <8 x i8> %op2 to <8 x i16>
  %mul = mul <8 x i16> %1, %2
  %shr = lshr <8 x i16> %mul, %splat
  %res = trunc <8 x i16> %shr to <8 x i8>
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @umulh_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: umulh_v16i8:
; CHECK: umull2 v2.8h, v0.16b, v1.16b
; CHECK: umull v0.8h, v0.8b, v1.8b
; CHECK: uzp2 v0.16b, v0.16b, v2.16b
; CHECK: ret
  %insert = insertelement <16 x i16> undef, i16 8, i64 0
  %splat = shufflevector <16 x i16> %insert, <16 x i16> undef, <16 x i32> zeroinitializer
  %1 = zext <16 x i8> %op1 to <16 x i16>
  %2 = zext <16 x i8> %op2 to <16 x i16>
  %mul = mul <16 x i16> %1, %2
  %shr = lshr <16 x i16> %mul, %splat
  %res = trunc <16 x i16> %shr to <16 x i8>
  ret <16 x i8> %res
}

define void @umulh_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: umulh_v32i8:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; VBITS_EQ_256-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256: umulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_256: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,32)]]
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBIGS_GE_512: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_512: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %insert = insertelement <32 x i16> undef, i16 8, i64 0
  %splat = shufflevector <32 x i16> %insert, <32 x i16> undef, <32 x i32> zeroinitializer
  %1 = zext <32 x i8> %op1 to <32 x i16>
  %2 = zext <32 x i8> %op2 to <32 x i16>
  %mul = mul <32 x i16> %1, %2
  %shr = lshr <32 x i16> %mul, %splat
  %res = trunc <32 x i16> %shr to <32 x i8>
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @umulh_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: umulh_v64i8:
; VBITS_EQ_512: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; VBITS_EQ_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_512: umulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512: ret

; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBIGS_GE_1024: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_1024: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %insert = insertelement <64 x i16> undef, i16 8, i64 0
  %splat = shufflevector <64 x i16> %insert, <64 x i16> undef, <64 x i32> zeroinitializer
  %1 = zext <64 x i8> %op1 to <64 x i16>
  %2 = zext <64 x i8> %op2 to <64 x i16>
  %mul = mul <64 x i16> %1, %2
  %shr = lshr <64 x i16> %mul, %splat
  %res = trunc <64 x i16> %shr to <64 x i8>
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @umulh_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: umulh_v128i8:
; VBITS_EQ_1024: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; VBITS_EQ_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_1024: umulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,128)]]
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBIGS_GE_2048: lsr [[RES]].h, [[PG]]/m, [[RES]].h, #8
; VBITS_GE_2048: st1b { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %insert = insertelement <128 x i16> undef, i16 8, i64 0
  %splat = shufflevector <128 x i16> %insert, <128 x i16> undef, <128 x i32> zeroinitializer
  %1 = zext <128 x i8> %op1 to <128 x i16>
  %2 = zext <128 x i8> %op2 to <128 x i16>
  %mul = mul <128 x i16> %1, %2
  %shr = lshr <128 x i16> %mul, %splat
  %res = trunc <128 x i16> %shr to <128 x i8>
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @umulh_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: umulh_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048: umulh [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %insert = insertelement <256 x i16> undef, i16 8, i64 0
  %splat = shufflevector <256 x i16> %insert, <256 x i16> undef, <256 x i32> zeroinitializer
  %1 = zext <256 x i8> %op1 to <256 x i16>
  %2 = zext <256 x i8> %op2 to <256 x i16>
  %mul = mul <256 x i16> %1, %2
  %shr = lshr <256 x i16> %mul, %splat
  %res = trunc <256 x i16> %shr to <256 x i8>
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @umulh_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: umulh_v4i16:
; CHECK: umull v0.4s, v0.4h, v1.4h
; CHECK: ushr v0.4s, v0.4s, #16
; CHECK: mov w8, v0.s[1]
; CHECK: mov w9, v0.s[2]
; CHECK: mov w10, v0.s[3]
; CHECK: mov v0.h[1], w8
; CHECK: mov v0.h[2], w9
; CHECK: mov v0.h[3], w10
; CHECK: ret
  %insert = insertelement <4 x i32> undef, i32 16, i64 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %1 = zext <4 x i16> %op1 to <4 x i32>
  %2 = zext <4 x i16> %op2 to <4 x i32>
  %mul = mul <4 x i32> %1, %2
  %shr = lshr <4 x i32> %mul, %splat
  %res = trunc <4 x i32> %shr to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @umulh_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: umulh_v8i16:
; CHECK: umull2 v2.4s, v0.8h, v1.8h
; CHECK: umull v0.4s, v0.4h, v1.4h
; CHECK: uzp2 v0.8h, v0.8h, v2.8h
; CHECK: ret
  %insert = insertelement <8 x i32> undef, i32 16, i64 0
  %splat = shufflevector <8 x i32> %insert, <8 x i32> undef, <8 x i32> zeroinitializer
  %1 = zext <8 x i16> %op1 to <8 x i32>
  %2 = zext <8 x i16> %op2 to <8 x i32>
  %mul = mul <8 x i32> %1, %2
  %shr = lshr <8 x i32> %mul, %splat
  %res = trunc <8 x i32> %shr to <8 x i16>
  ret <8 x i16> %res
}

define void @umulh_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: umulh_v16i16:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,16)]]
; VBITS_EQ_256-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256: umulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_256: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,16)]]
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_512: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %insert = insertelement <16 x i32> undef, i32 16, i64 0
  %splat = shufflevector <16 x i32> %insert, <16 x i32> undef, <16 x i32> zeroinitializer
  %1 = zext <16 x i16> %op1 to <16 x i32>
  %2 = zext <16 x i16> %op2 to <16 x i32>
  %mul = mul <16 x i32> %1, %2
  %shr = lshr <16 x i32> %mul, %splat
  %res = trunc <16 x i32> %shr to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @umulh_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: umulh_v32i16:
; VBITS_EQ_512: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,32)]]
; VBITS_EQ_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_512: umulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_512: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_512: ret

; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,32)]]
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_1024: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %insert = insertelement <32 x i32> undef, i32 16, i64 0
  %splat = shufflevector <32 x i32> %insert, <32 x i32> undef, <32 x i32> zeroinitializer
  %1 = zext <32 x i16> %op1 to <32 x i32>
  %2 = zext <32 x i16> %op2 to <32 x i32>
  %mul = mul <32 x i32> %1, %2
  %shr = lshr <32 x i32> %mul, %splat
  %res = trunc <32 x i32> %shr to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @umulh_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: umulh_v64i16:
; VBITS_EQ_1024: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,64)]]
; VBITS_EQ_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_1024: umulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,64)]]
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048: lsr [[RES]].s, [[PG]]/m, [[RES]].s, #16
; VBITS_GE_2048: st1h { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %insert = insertelement <64 x i32> undef, i32 16, i64 0
  %splat = shufflevector <64 x i32> %insert, <64 x i32> undef, <64 x i32> zeroinitializer
  %1 = zext <64 x i16> %op1 to <64 x i32>
  %2 = zext <64 x i16> %op2 to <64 x i32>
  %mul = mul <64 x i32> %1, %2
  %shr = lshr <64 x i32> %mul, %splat
  %res = trunc <64 x i32> %shr to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @umulh_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: umulh_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl[[#min(VBYTES,128)]]
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048: umulh [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %insert = insertelement <128 x i32> undef, i32 16, i64 0
  %splat = shufflevector <128 x i32> %insert, <128 x i32> undef, <128 x i32> zeroinitializer
  %1 = zext <128 x i16> %op1 to <128 x i32>
  %2 = zext <128 x i16> %op2 to <128 x i32>
  %mul = mul <128 x i32> %1, %2
  %shr = lshr <128 x i32> %mul, %splat
  %res = trunc <128 x i32> %shr to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i32> @umulh_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: umulh_v2i32:
; CHECK: ushll v0.2d, v0.2s, #0
; CHECK: ushll v1.2d, v1.2s, #0
; CHECK: ptrue p0.d, vl2
; CHECK: mul z0.d, p0/m, z0.d, z1.d
; CHECK: shrn v0.2s, v0.2d, #32
; CHECK: ret
  %insert = insertelement <2 x i64> undef, i64 32, i64 0
  %splat = shufflevector <2 x i64> %insert, <2 x i64> undef, <2 x i32> zeroinitializer
  %1 = zext <2 x i32> %op1 to <2 x i64>
  %2 = zext <2 x i32> %op2 to <2 x i64>
  %mul = mul <2 x i64> %1, %2
  %shr = lshr <2 x i64> %mul, %splat
  %res = trunc <2 x i64> %shr to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @umulh_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: umulh_v4i32:
; CHECK: umull2 v2.2d, v0.4s, v1.4s
; CHECK: umull v0.2d, v0.2s, v1.2s
; CHECK: uzp2 v0.4s, v0.4s, v2.4s
; CHECK: ret
  %insert = insertelement <4 x i64> undef, i64 32, i64 0
  %splat = shufflevector <4 x i64> %insert, <4 x i64> undef, <4 x i32> zeroinitializer
  %1 = zext <4 x i32> %op1 to <4 x i64>
  %2 = zext <4 x i32> %op2 to <4 x i64>
  %mul = mul <4 x i64> %1, %2
  %shr = lshr <4 x i64> %mul, %splat
  %res = trunc <4 x i64> %shr to <4 x i32>
  ret <4 x i32> %res
}

define void @umulh_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: umulh_v8i32:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,8)]]
; VBITS_EQ_256-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256: umulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_256: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_256: ret

; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,8)]]
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512: lsr [[RES]].d, [[PG]]/m, [[RES]].d, #32
; VBITS_GE_512: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %insert = insertelement <8 x i64> undef, i64 32, i64 0
  %splat = shufflevector <8 x i64> %insert, <8 x i64> undef, <8 x i32> zeroinitializer
  %1 = zext <8 x i32> %op1 to <8 x i64>
  %2 = zext <8 x i32> %op2 to <8 x i64>
  %mul = mul <8 x i64> %1, %2
  %shr = lshr <8 x i64> %mul, %splat
  %res = trunc <8 x i64> %shr to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @umulh_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: umulh_v16i32:
; VBITS_EQ_512: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,16)]]
; VBITS_EQ_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_512: umulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_512: st1w { [[RES]].s }, [[PG]], [x0]

; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,16)]]
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024: lsr [[RES]].d, [[PG]]/m, [[RES]].d, #32
; VBITS_GE_1024: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %insert = insertelement <16 x i64> undef, i64 32, i64 0
  %splat = shufflevector <16 x i64> %insert, <16 x i64> undef, <16 x i32> zeroinitializer
  %1 = zext <16 x i32> %op1 to <16 x i64>
  %2 = zext <16 x i32> %op2 to <16 x i64>
  %mul = mul <16 x i64> %1, %2
  %shr = lshr <16 x i64> %mul, %splat
  %res = trunc <16 x i64> %shr to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @umulh_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: umulh_v32i32:
; VBITS_EQ_1024: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,32)]]
; VBITS_EQ_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_1024: umulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_EQ_1024: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_1024: ret

; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,32)]]
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048: lsr [[RES]].d, [[PG]]/m, [[RES]].d, #32
; VBITS_GE_2048: st1w { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %insert = insertelement <32 x i64> undef, i64 32, i64 0
  %splat = shufflevector <32 x i64> %insert, <32 x i64> undef, <32 x i32> zeroinitializer
  %1 = zext <32 x i32> %op1 to <32 x i64>
  %2 = zext <32 x i32> %op2 to <32 x i64>
  %mul = mul <32 x i64> %1, %2
  %shr = lshr <32 x i64> %mul, %splat
  %res = trunc <32 x i64> %shr to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @umulh_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: umulh_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl[[#min(VBYTES,64)]]
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048: umulh [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %insert = insertelement <64 x i64> undef, i64 32, i64 0
  %splat = shufflevector <64 x i64> %insert, <64 x i64> undef, <64 x i32> zeroinitializer
  %1 = zext <64 x i32> %op1 to <64 x i64>
  %2 = zext <64 x i32> %op2 to <64 x i64>
  %mul = mul <64 x i64> %1, %2
  %shr = lshr <64 x i64> %mul, %splat
  %res = trunc <64 x i64> %shr to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <1 x i64> @umulh_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: umulh_v1i64:
; CHECK: ptrue p0.d, vl1
; CHECK: umulh z0.d, p0/m, z0.d, z1.d
; CHECK: ret
  %insert = insertelement <1 x i128> undef, i128 64, i128 0
  %splat = shufflevector <1 x i128> %insert, <1 x i128> undef, <1 x i32> zeroinitializer
  %1 = zext <1 x i64> %op1 to <1 x i128>
  %2 = zext <1 x i64> %op2 to <1 x i128>
  %mul = mul <1 x i128> %1, %2
  %shr = lshr <1 x i128> %mul, %splat
  %res = trunc <1 x i128> %shr to <1 x i64>
  ret <1 x i64> %res
}

; Vector i64 multiplications are not legal for NEON so use SVE when available.
define <2 x i64> @umulh_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: umulh_v2i64:
; CHECK: ptrue p0.d, vl2
; CHECK: umulh z0.d, p0/m, z0.d, z1.d
; CHECK: ret
  %insert = insertelement <2 x i128> undef, i128 64, i128 0
  %splat = shufflevector <2 x i128> %insert, <2 x i128> undef, <2 x i32> zeroinitializer
  %1 = zext <2 x i64> %op1 to <2 x i128>
  %2 = zext <2 x i64> %op2 to <2 x i128>
  %mul = mul <2 x i128> %1, %2
  %shr = lshr <2 x i128> %mul, %splat
  %res = trunc <2 x i128> %shr to <2 x i64>
  ret <2 x i64> %res
}

define void @umulh_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: umulh_v4i64:
; VBITS_GE_256: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,4)]]
; VBITS_GE_256-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_256-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_256: umulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_256: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_256: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %insert = insertelement <4 x i128> undef, i128 64, i128 0
  %splat = shufflevector <4 x i128> %insert, <4 x i128> undef, <4 x i32> zeroinitializer
  %1 = zext <4 x i64> %op1 to <4 x i128>
  %2 = zext <4 x i64> %op2 to <4 x i128>
  %mul = mul <4 x i128> %1, %2
  %shr = lshr <4 x i128> %mul, %splat
  %res = trunc <4 x i128> %shr to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @umulh_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: umulh_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,8)]]
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512: umulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %insert = insertelement <8 x i128> undef, i128 64, i128 0
  %splat = shufflevector <8 x i128> %insert, <8 x i128> undef, <8 x i32> zeroinitializer
  %1 = zext <8 x i64> %op1 to <8 x i128>
  %2 = zext <8 x i64> %op2 to <8 x i128>
  %mul = mul <8 x i128> %1, %2
  %shr = lshr <8 x i128> %mul, %splat
  %res = trunc <8 x i128> %shr to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @umulh_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: umulh_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,16)]]
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024: umulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %insert = insertelement <16 x i128> undef, i128 64, i128 0
  %splat = shufflevector <16 x i128> %insert, <16 x i128> undef, <16 x i32> zeroinitializer
  %1 = zext <16 x i64> %op1 to <16 x i128>
  %2 = zext <16 x i64> %op2 to <16 x i128>
  %mul = mul <16 x i128> %1, %2
  %shr = lshr <16 x i128> %mul, %splat
  %res = trunc <16 x i128> %shr to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @umulh_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: umulh_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl[[#min(VBYTES,32)]]
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048: umulh [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %insert = insertelement <32 x i128> undef, i128 64, i128 0
  %splat = shufflevector <32 x i128> %insert, <32 x i128> undef, <32 x i32> zeroinitializer
  %1 = zext <32 x i64> %op1 to <32 x i128>
  %2 = zext <32 x i64> %op2 to <32 x i128>
  %mul = mul <32 x i128> %1, %2
  %shr = lshr <32 x i128> %mul, %splat
  %res = trunc <32 x i128> %shr to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}
attributes #0 = { "target-features"="+sve" }
