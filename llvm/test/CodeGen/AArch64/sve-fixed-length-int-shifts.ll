; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; ASHR
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @ashr_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: ashr_v8i8:
; CHECK: neg v1.8b, v1.8b
; CHECK-NEXT: sshl v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = ashr <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @ashr_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: ashr_v16i8:
; CHECK: neg v1.16b, v1.16b
; CHECK-NEXT: sshl v0.16b, v0.16b, v1.16b
; CHECK-NEXT: ret
  %res = ashr <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @ashr_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: ashr_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: asr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = ashr <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @ashr_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: ashr_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: asr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[OFFSET_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: asr [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: asr [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0, x[[OFFSET_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = ashr <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @ashr_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: ashr_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: asr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = ashr <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @ashr_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: ashr_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: asr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = ashr <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @ashr_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: ashr_v4i16:
; CHECK: neg v1.4h, v1.4h
; CHECK-NEXT: sshl v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = ashr <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @ashr_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: ashr_v8i16:
; CHECK: neg v1.8h, v1.8h
; CHECK-NEXT: sshl v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = ashr <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @ashr_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: ashr_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: asr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = ashr <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @ashr_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: ashr_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: asr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1h { [[OP1_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP1_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1h { [[OP2_LO:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1h { [[OP2_HI:z[0-9]+]].h }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: asr [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: asr [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = ashr <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @ashr_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: ashr_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: asr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = ashr <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @ashr_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: ashr_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: asr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = ashr <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @ashr_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: ashr_v2i32:
; CHECK: neg v1.2s, v1.2s
; CHECK-NEXT: sshl v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = ashr <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @ashr_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: ashr_v4i32:
; CHECK: neg v1.4s, v1.4s
; CHECK-NEXT: sshl v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = ashr <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @ashr_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: ashr_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: asr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = ashr <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @ashr_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: ashr_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: asr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1w { [[OP1_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP1_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1w { [[OP2_LO:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1w { [[OP2_HI:z[0-9]+]].s }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: asr [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: asr [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = ashr <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @ashr_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: ashr_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: asr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = ashr <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @ashr_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: ashr_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: asr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = ashr <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @ashr_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: ashr_v1i64:
; CHECK: neg d1, d1
; CHECK-NEXT: sshl d0, d0, d1
; CHECK-NEXT: ret
  %res = ashr <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @ashr_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: ashr_v2i64:
; CHECK: neg v1.2d, v1.2d
; CHECK-NEXT: sshl v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %res = ashr <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @ashr_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: ashr_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: asr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = ashr <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @ashr_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: ashr_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: asr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1d { [[OP1_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP1_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1d { [[OP2_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[OP2_HI:z[0-9]+]].d }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: asr [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: asr [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = ashr <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @ashr_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: ashr_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: asr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = ashr <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @ashr_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: ashr_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: asr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = ashr <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; LSHR
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @lshr_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: lshr_v8i8:
; CHECK: neg v1.8b, v1.8b
; CHECK-NEXT: ushl v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = lshr <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @lshr_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: lshr_v16i8:
; CHECK: neg v1.16b, v1.16b
; CHECK-NEXT: ushl v0.16b, v0.16b, v1.16b
; CHECK-NEXT: ret
  %res = lshr <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @lshr_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: lshr_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: lsr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = lshr <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @lshr_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: lshr_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[OFFSET_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: lsr [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: lsr [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0, x[[OFFSET_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = lshr <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @lshr_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: lshr_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = lshr <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @lshr_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: lshr_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsr [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = lshr <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @lshr_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: lshr_v4i16:
; CHECK: neg v1.4h, v1.4h
; CHECK-NEXT: ushl v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = lshr <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @lshr_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: lshr_v8i16:
; CHECK: neg v1.8h, v1.8h
; CHECK-NEXT: ushl v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = lshr <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @lshr_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: lshr_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: lsr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = lshr <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @lshr_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: lshr_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1h { [[OP1_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP1_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1h { [[OP2_LO:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1h { [[OP2_HI:z[0-9]+]].h }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsr [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: lsr [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = lshr <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @lshr_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: lshr_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = lshr <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @lshr_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: lshr_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsr [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = lshr <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @lshr_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: lshr_v2i32:
; CHECK: neg v1.2s, v1.2s
; CHECK-NEXT: ushl v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = lshr <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @lshr_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: lshr_v4i32:
; CHECK: neg v1.4s, v1.4s
; CHECK-NEXT: ushl v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = lshr <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @lshr_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: lshr_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: lsr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = lshr <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @lshr_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: lshr_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1w { [[OP1_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP1_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1w { [[OP2_LO:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1w { [[OP2_HI:z[0-9]+]].s }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsr [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: lsr [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = lshr <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @lshr_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: lshr_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = lshr <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @lshr_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: lshr_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsr [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = lshr <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @lshr_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: lshr_v1i64:
; CHECK: neg d1, d1
; CHECK-NEXT: ushl d0, d0, d1
; CHECK-NEXT: ret
  %res = lshr <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @lshr_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: lshr_v2i64:
; CHECK: neg v1.2d, v1.2d
; CHECK-NEXT: ushl v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %res = lshr <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @lshr_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: lshr_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: lsr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = lshr <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @lshr_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: lshr_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1d { [[OP1_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP1_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1d { [[OP2_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[OP2_HI:z[0-9]+]].d }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsr [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: lsr [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = lshr <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @lshr_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: lshr_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = lshr <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @lshr_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: lshr_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsr [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = lshr <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; SHL
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @shl_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: shl_v8i8:
; CHECK: ushl v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = shl <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @shl_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: shl_v16i8:
; CHECK: ushl v0.16b, v0.16b, v1.16b
; CHECK-NEXT: ret
  %res = shl <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @shl_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: shl_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: lsl [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = shl <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @shl_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: shl_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsl [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[OFFSET_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFFSET_HI]]]
; VBITS_EQ_256-DAG: lsl [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: lsl [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0, x[[OFFSET_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = shl <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @shl_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: shl_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsl [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = shl <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @shl_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: shl_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsl [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = shl <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @shl_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: shl_v4i16:
; CHECK: ushl v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = shl <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @shl_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: shl_v8i16:
; CHECK: ushl v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = shl <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @shl_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: shl_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: lsl [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = shl <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @shl_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: shl_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsl [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1h { [[OP1_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP1_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1h { [[OP2_LO:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1h { [[OP2_HI:z[0-9]+]].h }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsl [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: lsl [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = shl <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @shl_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: shl_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsl [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = shl <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @shl_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: shl_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsl [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = shl <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @shl_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: shl_v2i32:
; CHECK: ushl v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = shl <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @shl_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: shl_v4i32:
; CHECK: ushl v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = shl <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @shl_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: shl_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: lsl [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = shl <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @shl_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: shl_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsl [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1w { [[OP1_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP1_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1w { [[OP2_LO:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1w { [[OP2_HI:z[0-9]+]].s }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsl [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: lsl [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = shl <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @shl_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: shl_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsl [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = shl <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @shl_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: shl_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsl [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = shl <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @shl_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: shl_v1i64:
; CHECK: ushl d0, d0, d1
; CHECK-NEXT: ret
  %res = shl <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @shl_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: shl_v2i64:
; CHECK: ushl v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %res = shl <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @shl_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: shl_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: lsl [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = shl <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @shl_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: shl_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: lsl [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1d { [[OP1_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP1_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1d { [[OP2_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[OP2_HI:z[0-9]+]].d }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: lsl [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: lsl [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = shl <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @shl_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: shl_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: lsl [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = shl <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @shl_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: shl_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: lsl [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = shl <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
