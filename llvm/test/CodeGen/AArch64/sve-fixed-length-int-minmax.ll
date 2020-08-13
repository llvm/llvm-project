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
; SMAX
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @smax_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: smax_v8i8:
; CHECK: smax v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = call <8 x i8> @llvm.smax.v8i8(<8 x i8> %op1, <8 x i8> %op2)
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @smax_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: smax_v16i8:
; CHECK: smax v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %op1, <16 x i8> %op2)
  ret <16 x i8> %res
}

define void @smax_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: smax_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: smax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = call <32 x i8> @llvm.smax.v32i8(<32 x i8> %op1, <32 x i8> %op2)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @smax_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: smax_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
;
; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: smax [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: smax [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0, x[[A]]]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = call <64 x i8> @llvm.smax.v64i8(<64 x i8> %op1, <64 x i8> %op2)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @smax_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: smax_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = call <128 x i8> @llvm.smax.v128i8(<128 x i8> %op1, <128 x i8> %op2)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @smax_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: smax_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = call <256 x i8> @llvm.smax.v256i8(<256 x i8> %op1, <256 x i8> %op2)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @smax_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: smax_v4i16:
; CHECK: smax v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x i16> @llvm.smax.v4i16(<4 x i16> %op1, <4 x i16> %op2)
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @smax_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: smax_v8i16:
; CHECK: smax v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x i16> @llvm.smax.v8i16(<8 x i16> %op1, <8 x i16> %op2)
  ret <8 x i16> %res
}

define void @smax_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: smax_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: smax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = call <16 x i16> @llvm.smax.v16i16(<16 x i16> %op1, <16 x i16> %op2)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @smax_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: smax_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: smax [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: smax [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = call <32 x i16> @llvm.smax.v32i16(<32 x i16> %op1, <32 x i16> %op2)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @smax_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: smax_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = call <64 x i16> @llvm.smax.v64i16(<64 x i16> %op1, <64 x i16> %op2)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @smax_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: smax_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = call <128 x i16> @llvm.smax.v128i16(<128 x i16> %op1, <128 x i16> %op2)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @smax_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: smax_v2i32:
; CHECK: smax v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x i32> @llvm.smax.v2i32(<2 x i32> %op1, <2 x i32> %op2)
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @smax_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: smax_v4i32:
; CHECK: smax v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %op1, <4 x i32> %op2)
  ret <4 x i32> %res
}

define void @smax_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: smax_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: smax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %op1, <8 x i32> %op2)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @smax_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: smax_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: smax [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: smax [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = call <16 x i32> @llvm.smax.v16i32(<16 x i32> %op1, <16 x i32> %op2)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @smax_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: smax_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = call <32 x i32> @llvm.smax.v32i32(<32 x i32> %op1, <32 x i32> %op2)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @smax_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: smax_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = call <64 x i32> @llvm.smax.v64i32(<64 x i32> %op1, <64 x i32> %op2)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 max are not legal for NEON so use SVE when available.
define <1 x i64> @smax_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: smax_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: smax z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <1 x i64> @llvm.smax.v1i64(<1 x i64> %op1, <1 x i64> %op2)
  ret <1 x i64> %res
}

; Vector i64 max are not legal for NEON so use SVE when available.
define <2 x i64> @smax_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: smax_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: smax z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %op1, <2 x i64> %op2)
  ret <2 x i64> %res
}

define void @smax_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: smax_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: smax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = call <4 x i64> @llvm.smax.v4i64(<4 x i64> %op1, <4 x i64> %op2)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @smax_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: smax_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: smax [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: smax [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = call <8 x i64> @llvm.smax.v8i64(<8 x i64> %op1, <8 x i64> %op2)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @smax_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: smax_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = call <16 x i64> @llvm.smax.v16i64(<16 x i64> %op1, <16 x i64> %op2)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @smax_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: smax_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = call <32 x i64> @llvm.smax.v32i64(<32 x i64> %op1, <32 x i64> %op2)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; SMIN
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @smin_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: smin_v8i8:
; CHECK: smin v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = call <8 x i8> @llvm.smin.v8i8(<8 x i8> %op1, <8 x i8> %op2)
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @smin_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: smin_v16i8:
; CHECK: smin v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %op1, <16 x i8> %op2)
  ret <16 x i8> %res
}

define void @smin_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: smin_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: smin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = call <32 x i8> @llvm.smin.v32i8(<32 x i8> %op1, <32 x i8> %op2)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @smin_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: smin_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
;
; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: smin [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: smin [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0, x[[A]]]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0]
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = call <64 x i8> @llvm.smin.v64i8(<64 x i8> %op1, <64 x i8> %op2)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @smin_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: smin_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = call <128 x i8> @llvm.smin.v128i8(<128 x i8> %op1, <128 x i8> %op2)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @smin_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: smin_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = call <256 x i8> @llvm.smin.v256i8(<256 x i8> %op1, <256 x i8> %op2)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @smin_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: smin_v4i16:
; CHECK: smin v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x i16> @llvm.smin.v4i16(<4 x i16> %op1, <4 x i16> %op2)
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @smin_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: smin_v8i16:
; CHECK: smin v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x i16> @llvm.smin.v8i16(<8 x i16> %op1, <8 x i16> %op2)
  ret <8 x i16> %res
}

define void @smin_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: smin_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: smin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = call <16 x i16> @llvm.smin.v16i16(<16 x i16> %op1, <16 x i16> %op2)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @smin_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: smin_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: smin [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: smin [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = call <32 x i16> @llvm.smin.v32i16(<32 x i16> %op1, <32 x i16> %op2)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @smin_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: smin_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = call <64 x i16> @llvm.smin.v64i16(<64 x i16> %op1, <64 x i16> %op2)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @smin_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: smin_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = call <128 x i16> @llvm.smin.v128i16(<128 x i16> %op1, <128 x i16> %op2)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @smin_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: smin_v2i32:
; CHECK: smin v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x i32> @llvm.smin.v2i32(<2 x i32> %op1, <2 x i32> %op2)
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @smin_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: smin_v4i32:
; CHECK: smin v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %op1, <4 x i32> %op2)
  ret <4 x i32> %res
}

define void @smin_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: smin_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: smin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %op1, <8 x i32> %op2)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @smin_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: smin_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: smin [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: smin [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = call <16 x i32> @llvm.smin.v16i32(<16 x i32> %op1, <16 x i32> %op2)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @smin_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: smin_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = call <32 x i32> @llvm.smin.v32i32(<32 x i32> %op1, <32 x i32> %op2)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @smin_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: smin_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = call <64 x i32> @llvm.smin.v64i32(<64 x i32> %op1, <64 x i32> %op2)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 min are not legal for NEON so use SVE when available.
define <1 x i64> @smin_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: smin_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: smin z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <1 x i64> @llvm.smin.v1i64(<1 x i64> %op1, <1 x i64> %op2)
  ret <1 x i64> %res
}

; Vector i64 min are not legal for NEON so use SVE when available.
define <2 x i64> @smin_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: smin_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: smin z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %op1, <2 x i64> %op2)
  ret <2 x i64> %res
}

define void @smin_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: smin_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: smin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = call <4 x i64> @llvm.smin.v4i64(<4 x i64> %op1, <4 x i64> %op2)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @smin_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: smin_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: smin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: smin [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: smin [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = call <8 x i64> @llvm.smin.v8i64(<8 x i64> %op1, <8 x i64> %op2)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @smin_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: smin_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: smin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = call <16 x i64> @llvm.smin.v16i64(<16 x i64> %op1, <16 x i64> %op2)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @smin_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: smin_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: smin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = call <32 x i64> @llvm.smin.v32i64(<32 x i64> %op1, <32 x i64> %op2)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UMAX
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @umax_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: umax_v8i8:
; CHECK: umax v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = call <8 x i8> @llvm.umax.v8i8(<8 x i8> %op1, <8 x i8> %op2)
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @umax_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: umax_v16i8:
; CHECK: umax v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %op1, <16 x i8> %op2)
  ret <16 x i8> %res
}

define void @umax_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: umax_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: umax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = call <32 x i8> @llvm.umax.v32i8(<32 x i8> %op1, <32 x i8> %op2)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @umax_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: umax_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
;
; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: umax [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: umax [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0, x[[A]]]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = call <64 x i8> @llvm.umax.v64i8(<64 x i8> %op1, <64 x i8> %op2)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @umax_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: umax_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = call <128 x i8> @llvm.umax.v128i8(<128 x i8> %op1, <128 x i8> %op2)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @umax_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: umax_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umax [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = call <256 x i8> @llvm.umax.v256i8(<256 x i8> %op1, <256 x i8> %op2)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @umax_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: umax_v4i16:
; CHECK: umax v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x i16> @llvm.umax.v4i16(<4 x i16> %op1, <4 x i16> %op2)
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @umax_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: umax_v8i16:
; CHECK: umax v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x i16> @llvm.umax.v8i16(<8 x i16> %op1, <8 x i16> %op2)
  ret <8 x i16> %res
}

define void @umax_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: umax_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: umax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = call <16 x i16> @llvm.umax.v16i16(<16 x i16> %op1, <16 x i16> %op2)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @umax_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: umax_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: umax [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: umax [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = call <32 x i16> @llvm.umax.v32i16(<32 x i16> %op1, <32 x i16> %op2)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @umax_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: umax_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = call <64 x i16> @llvm.umax.v64i16(<64 x i16> %op1, <64 x i16> %op2)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @umax_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: umax_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umax [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = call <128 x i16> @llvm.umax.v128i16(<128 x i16> %op1, <128 x i16> %op2)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @umax_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: umax_v2i32:
; CHECK: umax v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x i32> @llvm.umax.v2i32(<2 x i32> %op1, <2 x i32> %op2)
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @umax_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: umax_v4i32:
; CHECK: umax v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %op1, <4 x i32> %op2)
  ret <4 x i32> %res
}

define void @umax_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: umax_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: umax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %op1, <8 x i32> %op2)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @umax_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: umax_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: umax [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: umax [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = call <16 x i32> @llvm.umax.v16i32(<16 x i32> %op1, <16 x i32> %op2)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @umax_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: umax_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = call <32 x i32> @llvm.umax.v32i32(<32 x i32> %op1, <32 x i32> %op2)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @umax_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: umax_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umax [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = call <64 x i32> @llvm.umax.v64i32(<64 x i32> %op1, <64 x i32> %op2)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 max are not legal for NEON so use SVE when available.
define <1 x i64> @umax_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: umax_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: umax z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <1 x i64> @llvm.umax.v1i64(<1 x i64> %op1, <1 x i64> %op2)
  ret <1 x i64> %res
}

; Vector i64 max are not legal for NEON so use SVE when available.
define <2 x i64> @umax_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: umax_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: umax z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %op1, <2 x i64> %op2)
  ret <2 x i64> %res
}

define void @umax_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: umax_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: umax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = call <4 x i64> @llvm.umax.v4i64(<4 x i64> %op1, <4 x i64> %op2)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @umax_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: umax_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: umax [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: umax [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = call <8 x i64> @llvm.umax.v8i64(<8 x i64> %op1, <8 x i64> %op2)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @umax_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: umax_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = call <16 x i64> @llvm.umax.v16i64(<16 x i64> %op1, <16 x i64> %op2)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @umax_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: umax_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umax [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = call <32 x i64> @llvm.umax.v32i64(<32 x i64> %op1, <32 x i64> %op2)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UMIN
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @umin_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: umin_v8i8:
; CHECK: umin v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = call <8 x i8> @llvm.umin.v8i8(<8 x i8> %op1, <8 x i8> %op2)
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @umin_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: umin_v16i8:
; CHECK: umin v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %op1, <16 x i8> %op2)
  ret <16 x i8> %res
}

define void @umin_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: umin_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: umin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = call <32 x i8> @llvm.umin.v32i8(<32 x i8> %op1, <32 x i8> %op2)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @umin_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: umin_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
;
; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[A:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1, x[[A]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: umin [[RES_LO:z[0-9]+]].b, [[PG]]/m, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: umin [[RES_HI:z[0-9]+]].b, [[PG]]/m, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: st1b { [[RES_LO]].b }, [[PG]], [x0, x[[A]]]
; VBITS_EQ_256-DAG: st1b { [[RES_HI]].b }, [[PG]], [x0]
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = call <64 x i8> @llvm.umin.v64i8(<64 x i8> %op1, <64 x i8> %op2)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @umin_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: umin_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = call <128 x i8> @llvm.umin.v128i8(<128 x i8> %op1, <128 x i8> %op2)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @umin_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: umin_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umin [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = call <256 x i8> @llvm.umin.v256i8(<256 x i8> %op1, <256 x i8> %op2)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @umin_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: umin_v4i16:
; CHECK: umin v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x i16> @llvm.umin.v4i16(<4 x i16> %op1, <4 x i16> %op2)
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @umin_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: umin_v8i16:
; CHECK: umin v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x i16> @llvm.umin.v8i16(<8 x i16> %op1, <8 x i16> %op2)
  ret <8 x i16> %res
}

define void @umin_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: umin_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: umin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = call <16 x i16> @llvm.umin.v16i16(<16 x i16> %op1, <16 x i16> %op2)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @umin_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: umin_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: umin [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: umin [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = call <32 x i16> @llvm.umin.v32i16(<32 x i16> %op1, <32 x i16> %op2)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @umin_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: umin_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = call <64 x i16> @llvm.umin.v64i16(<64 x i16> %op1, <64 x i16> %op2)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @umin_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: umin_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umin [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = call <128 x i16> @llvm.umin.v128i16(<128 x i16> %op1, <128 x i16> %op2)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @umin_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: umin_v2i32:
; CHECK: umin v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x i32> @llvm.umin.v2i32(<2 x i32> %op1, <2 x i32> %op2)
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @umin_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: umin_v4i32:
; CHECK: umin v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %op1, <4 x i32> %op2)
  ret <4 x i32> %res
}

define void @umin_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: umin_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: umin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %op1, <8 x i32> %op2)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @umin_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: umin_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: umin [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: umin [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = call <16 x i32> @llvm.umin.v16i32(<16 x i32> %op1, <16 x i32> %op2)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @umin_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: umin_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = call <32 x i32> @llvm.umin.v32i32(<32 x i32> %op1, <32 x i32> %op2)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @umin_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: umin_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umin [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = call <64 x i32> @llvm.umin.v64i32(<64 x i32> %op1, <64 x i32> %op2)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 min are not legal for NEON so use SVE when available.
define <1 x i64> @umin_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: umin_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: umin z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <1 x i64> @llvm.umin.v1i64(<1 x i64> %op1, <1 x i64> %op2)
  ret <1 x i64> %res
}

; Vector i64 min are not legal for NEON so use SVE when available.
define <2 x i64> @umin_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: umin_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: umin z0.d, [[PG]]/m, z0.d, z1.d
; CHECK-NEXT: ret
  %res = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %op1, <2 x i64> %op2)
  ret <2 x i64> %res
}

define void @umin_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: umin_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: umin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = call <4 x i64> @llvm.umin.v4i64(<4 x i64> %op1, <4 x i64> %op2)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @umin_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: umin_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: umin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: umin [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: umin [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = call <8 x i64> @llvm.umin.v8i64(<8 x i64> %op1, <8 x i64> %op2)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @umin_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: umin_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: umin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = call <16 x i64> @llvm.umin.v16i64(<16 x i64> %op1, <16 x i64> %op2)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @umin_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: umin_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: umin [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = call <32 x i64> @llvm.umin.v32i64(<32 x i64> %op1, <32 x i64> %op2)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

declare <8 x i8> @llvm.smin.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.smin.v16i8(<16 x i8>, <16 x i8>)
declare <32 x i8> @llvm.smin.v32i8(<32 x i8>, <32 x i8>)
declare <64 x i8> @llvm.smin.v64i8(<64 x i8>, <64 x i8>)
declare <128 x i8> @llvm.smin.v128i8(<128 x i8>, <128 x i8>)
declare <256 x i8> @llvm.smin.v256i8(<256 x i8>, <256 x i8>)
declare <4 x i16> @llvm.smin.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.smin.v8i16(<8 x i16>, <8 x i16>)
declare <16 x i16> @llvm.smin.v16i16(<16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.smin.v32i16(<32 x i16>, <32 x i16>)
declare <64 x i16> @llvm.smin.v64i16(<64 x i16>, <64 x i16>)
declare <128 x i16> @llvm.smin.v128i16(<128 x i16>, <128 x i16>)
declare <2 x i32> @llvm.smin.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.smin.v4i32(<4 x i32>, <4 x i32>)
declare <8 x i32> @llvm.smin.v8i32(<8 x i32>, <8 x i32>)
declare <16 x i32> @llvm.smin.v16i32(<16 x i32>, <16 x i32>)
declare <32 x i32> @llvm.smin.v32i32(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.smin.v64i32(<64 x i32>, <64 x i32>)
declare <1 x i64> @llvm.smin.v1i64(<1 x i64>, <1 x i64>)
declare <2 x i64> @llvm.smin.v2i64(<2 x i64>, <2 x i64>)
declare <4 x i64> @llvm.smin.v4i64(<4 x i64>, <4 x i64>)
declare <8 x i64> @llvm.smin.v8i64(<8 x i64>, <8 x i64>)
declare <16 x i64> @llvm.smin.v16i64(<16 x i64>, <16 x i64>)
declare <32 x i64> @llvm.smin.v32i64(<32 x i64>, <32 x i64>)

declare <8 x i8> @llvm.smax.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.smax.v16i8(<16 x i8>, <16 x i8>)
declare <32 x i8> @llvm.smax.v32i8(<32 x i8>, <32 x i8>)
declare <64 x i8> @llvm.smax.v64i8(<64 x i8>, <64 x i8>)
declare <128 x i8> @llvm.smax.v128i8(<128 x i8>, <128 x i8>)
declare <256 x i8> @llvm.smax.v256i8(<256 x i8>, <256 x i8>)
declare <4 x i16> @llvm.smax.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.smax.v8i16(<8 x i16>, <8 x i16>)
declare <16 x i16> @llvm.smax.v16i16(<16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.smax.v32i16(<32 x i16>, <32 x i16>)
declare <64 x i16> @llvm.smax.v64i16(<64 x i16>, <64 x i16>)
declare <128 x i16> @llvm.smax.v128i16(<128 x i16>, <128 x i16>)
declare <2 x i32> @llvm.smax.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>)
declare <8 x i32> @llvm.smax.v8i32(<8 x i32>, <8 x i32>)
declare <16 x i32> @llvm.smax.v16i32(<16 x i32>, <16 x i32>)
declare <32 x i32> @llvm.smax.v32i32(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.smax.v64i32(<64 x i32>, <64 x i32>)
declare <1 x i64> @llvm.smax.v1i64(<1 x i64>, <1 x i64>)
declare <2 x i64> @llvm.smax.v2i64(<2 x i64>, <2 x i64>)
declare <4 x i64> @llvm.smax.v4i64(<4 x i64>, <4 x i64>)
declare <8 x i64> @llvm.smax.v8i64(<8 x i64>, <8 x i64>)
declare <16 x i64> @llvm.smax.v16i64(<16 x i64>, <16 x i64>)
declare <32 x i64> @llvm.smax.v32i64(<32 x i64>, <32 x i64>)

declare <8 x i8> @llvm.umin.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.umin.v16i8(<16 x i8>, <16 x i8>)
declare <32 x i8> @llvm.umin.v32i8(<32 x i8>, <32 x i8>)
declare <64 x i8> @llvm.umin.v64i8(<64 x i8>, <64 x i8>)
declare <128 x i8> @llvm.umin.v128i8(<128 x i8>, <128 x i8>)
declare <256 x i8> @llvm.umin.v256i8(<256 x i8>, <256 x i8>)
declare <4 x i16> @llvm.umin.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.umin.v8i16(<8 x i16>, <8 x i16>)
declare <16 x i16> @llvm.umin.v16i16(<16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.umin.v32i16(<32 x i16>, <32 x i16>)
declare <64 x i16> @llvm.umin.v64i16(<64 x i16>, <64 x i16>)
declare <128 x i16> @llvm.umin.v128i16(<128 x i16>, <128 x i16>)
declare <2 x i32> @llvm.umin.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.umin.v4i32(<4 x i32>, <4 x i32>)
declare <8 x i32> @llvm.umin.v8i32(<8 x i32>, <8 x i32>)
declare <16 x i32> @llvm.umin.v16i32(<16 x i32>, <16 x i32>)
declare <32 x i32> @llvm.umin.v32i32(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.umin.v64i32(<64 x i32>, <64 x i32>)
declare <1 x i64> @llvm.umin.v1i64(<1 x i64>, <1 x i64>)
declare <2 x i64> @llvm.umin.v2i64(<2 x i64>, <2 x i64>)
declare <4 x i64> @llvm.umin.v4i64(<4 x i64>, <4 x i64>)
declare <8 x i64> @llvm.umin.v8i64(<8 x i64>, <8 x i64>)
declare <16 x i64> @llvm.umin.v16i64(<16 x i64>, <16 x i64>)
declare <32 x i64> @llvm.umin.v32i64(<32 x i64>, <32 x i64>)

declare <8 x i8> @llvm.umax.v8i8(<8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.umax.v16i8(<16 x i8>, <16 x i8>)
declare <32 x i8> @llvm.umax.v32i8(<32 x i8>, <32 x i8>)
declare <64 x i8> @llvm.umax.v64i8(<64 x i8>, <64 x i8>)
declare <128 x i8> @llvm.umax.v128i8(<128 x i8>, <128 x i8>)
declare <256 x i8> @llvm.umax.v256i8(<256 x i8>, <256 x i8>)
declare <4 x i16> @llvm.umax.v4i16(<4 x i16>, <4 x i16>)
declare <8 x i16> @llvm.umax.v8i16(<8 x i16>, <8 x i16>)
declare <16 x i16> @llvm.umax.v16i16(<16 x i16>, <16 x i16>)
declare <32 x i16> @llvm.umax.v32i16(<32 x i16>, <32 x i16>)
declare <64 x i16> @llvm.umax.v64i16(<64 x i16>, <64 x i16>)
declare <128 x i16> @llvm.umax.v128i16(<128 x i16>, <128 x i16>)
declare <2 x i32> @llvm.umax.v2i32(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.umax.v4i32(<4 x i32>, <4 x i32>)
declare <8 x i32> @llvm.umax.v8i32(<8 x i32>, <8 x i32>)
declare <16 x i32> @llvm.umax.v16i32(<16 x i32>, <16 x i32>)
declare <32 x i32> @llvm.umax.v32i32(<32 x i32>, <32 x i32>)
declare <64 x i32> @llvm.umax.v64i32(<64 x i32>, <64 x i32>)
declare <1 x i64> @llvm.umax.v1i64(<1 x i64>, <1 x i64>)
declare <2 x i64> @llvm.umax.v2i64(<2 x i64>, <2 x i64>)
declare <4 x i64> @llvm.umax.v4i64(<4 x i64>, <4 x i64>)
declare <8 x i64> @llvm.umax.v8i64(<8 x i64>, <8 x i64>)
declare <16 x i64> @llvm.umax.v16i64(<16 x i64>, <16 x i64>)
declare <32 x i64> @llvm.umax.v32i64(<32 x i64>, <32 x i64>)

