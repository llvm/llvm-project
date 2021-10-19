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
; FCVTZU H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x i16> @fcvtzu_v4f16_v4i16(<4 x half> %op1) #0 {
; CHECK-LABEL: fcvtzu_v4f16_v4i16:
; CHECK: fcvtzu v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = fptoui <4 x half> %op1 to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define void @fcvtzu_v8f16_v8i16(<8 x half>* %a, <8 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f16_v8i16:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: fcvtzu v0.8h, v0.8h
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptoui <8 x half> %op1 to <8 x i16>
  store <8 x i16> %res, <8 x i16>* %b
  ret void
}

define void @fcvtzu_v16f16_v16i16(<16 x half>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f16_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptoui <16 x half> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzu_v32f16_v32i16(<32 x half>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f16_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[LO]].h
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x1, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptoui <32 x half> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

define void @fcvtzu_v64f16_v64i16(<64 x half>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v64f16_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fptoui <64 x half> %op1 to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %b
  ret void
}

define void @fcvtzu_v128f16_v128i16(<128 x half>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v128f16_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %res = fptoui <128 x half> %op1 to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %b
  ret void
}

;
; FCVTZU H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x i32> @fcvtzu_v2f16_v2i32(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f16_v2i32:
; CHECK: fcvtl v0.4s, v0.4h
; CHECK-NEXT: fcvtzu v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptoui <2 x half> %op1 to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @fcvtzu_v4f16_v4i32(<4 x half> %op1) #0 {
; CHECK-LABEL: fcvtzu_v4f16_v4i32:
; CHECK: fcvtzu v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptoui <4 x half> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzu_v8f16_v8i32(<8 x half>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f16_v8i32:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG]]/m, [[UPK]].h
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptoui <8 x half> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzu_v16f16_v16i32(<16 x half>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f16_v16i32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[VEC:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].s, [[VEC]].h
; VBITS_EQ_256-DAG: ext [[VEC_HI:z[0-9]+]].b, [[VEC]].b, [[VEC]].b, #16
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].s, [[VEC_HI]].h
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].s, [[PG2]]/m, [[UPK_LO]].h
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].s, [[PG2]]/m, [[UPK_HI]].h
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG2]], [x1, x[[NUMELTS]], lsl #2]
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptoui <16 x half> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzu_v32f16_v32i32(<32 x half>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f16_v32i32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptoui <32 x half> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

define void @fcvtzu_v64f16_v64i32(<64 x half>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v64f16_v64i32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fptoui <64 x half> %op1 to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %b
  ret void
}

;
; FCVTZU H -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzu_v1f16_v1i64(<1 x half> %op1) #0 {
; CHECK-LABEL: fcvtzu_v1f16_v1i64:
; CHECK: fcvtzu x8, h0
; CHECK-NEXT: fmov d0, x8
; CHECK-NEXT: ret
  %res = fptoui <1 x half> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x i64> @fcvtzu_v2f16_v2i64(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f16_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvtzu z0.d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: ret
  %res = fptoui <2 x half> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzu_v4f16_v4i64(<4 x half>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v4f16_v4i64:
; CHECK: ldr d[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x half>, <4 x half>* %a
  %res = fptoui <4 x half> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzu_v8f16_v8i64(<8 x half>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f16_v8i64:
; VBITS_GE_512: ldr q[[OP:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ldr q[[OP:[0-9]+]], [x0]
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ext v[[HI:[0-9]+]].16b, v[[LO:[0-9]+]].16b, v[[OP]].16b, #8
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK2_LO]].h
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK2_HI]].h
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG2]], [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptoui <8 x half> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzu_v16f16_v16i64(<16 x half>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f16_v16i64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptoui <16 x half> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzu_v32f16_v32i64(<32 x half>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f16_v32i64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK]].s
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptoui <32 x half> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}

;
; FCVTZU S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x i16> @fcvtzu_v2f32_v2i16(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f32_v2i16:
; CHECK: fcvtzs v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = fptoui <2 x float> %op1 to <2 x i16>
  ret <2 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i16> @fcvtzu_v4f32_v4i16(<4 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v4f32_v4i16:
; CHECK: fcvtzu v1.4s, v0.4s
; CHECK-NEXT: mov w8, v1.s[1]
; CHECK-NEXT: mov w9, v1.s[2]
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: mov v0.h[1], w8
; CHECK-NEXT: mov w8, v1.s[3]
; CHECK-NEXT: mov v0.h[2], w9
; CHECK-NEXT: mov v0.h[3], w8
; CHECK-NEXT: ret
  %res = fptoui <4 x float> %op1 to <4 x i16>
  ret <4 x i16> %res
}

define <8 x i16> @fcvtzu_v8f32_v8i16(<8 x float>* %a) #0 {
; CHECK-LABEL: fcvtzu_v8f32_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: fcvtzu [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; CHECK-NEXT: uzp1 z0.h, [[CVT]].h, [[CVT]].h
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptoui <8 x float> %op1 to <8 x i16>
  ret <8 x i16> %res
}

define void @fcvtzu_v16f32_v16i16(<16 x float>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f32_v16i16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: fcvtzu [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].h, vl8
; VBITS_EQ_256-DAG: fcvtzu [[CVT_HI:z[0-9]+]].s, [[PG2]]/m, [[HI]].s
; VBITS_EQ_256-DAG: fcvtzu [[CVT_LO:z[0-9]+]].s, [[PG2]]/m, [[LO]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].h, [[CVT_LO]].h, [[CVT_LO]].h
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].h, [[CVT_HI]].h, [[CVT_HI]].h
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].h, [[PG3]], [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: st1h { [[RES]].h }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptoui <16 x float> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzu_v32f32_v32i16(<32 x float>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f32_v32i16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: fcvtzu [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptoui <32 x float> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

define void @fcvtzu_v64f32_v64i16(<64 x float>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v64f32_v64i16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptoui <64 x float> %op1 to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %b
  ret void
}

;
; FCVTZU S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x i32> @fcvtzu_v2f32_v2i32(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f32_v2i32:
; CHECK: fcvtzu v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = fptoui <2 x float> %op1 to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @fcvtzu_v4f32_v4i32(<4 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v4f32_v4i32:
; CHECK: fcvtzu v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptoui <4 x float> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzu_v8f32_v8i32(<8 x float>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f32_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptoui <8 x float> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzu_v16f32_v16i32(<16 x float>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f32_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[LO]].s
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x1, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptoui <16 x float> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzu_v32f32_v32i32(<32 x float>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f32_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptoui <32 x float> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

define void @fcvtzu_v64f32_v64i32(<64 x float>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v64f32_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptoui <64 x float> %op1 to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %b
  ret void
}

;
; FCVTZU S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzu_v1f32_v1i64(<1 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v1f32_v1i64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: fcvtzu v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <1 x float> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @fcvtzu_v2f32_v2i64(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f32_v2i64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: fcvtzu v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <2 x float> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzu_v4f32_v4i64(<4 x float>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v4f32_v4i64:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[OP]].s
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK]].s
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x float>, <4 x float>* %a
  %res = fptoui <4 x float> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzu_v8f32_v8i64(<8 x float>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f32_v8i64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG1]]/m, [[UPK]].s
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VEC:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].d, [[VEC]].s
; VBITS_EQ_256-DAG: ext [[VEC_HI:z[0-9]+]].b, [[VEC]].b, [[VEC]].b, #16
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].d, [[VEC_HI]].s
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK_LO]].s
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK_HI]].s
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG2]], [x1, x[[NUMELTS]], lsl #3]
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptoui <8 x float> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzu_v16f32_v16i64(<16 x float>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f32_v16i64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptoui <16 x float> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzu_v32f32_v32i64(<32 x float>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f32_v32i64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptoui <32 x float> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}


;
; FCVTZU D -> H
;

; v1f64 is perfered to be widened to v4f64, so use SVE
define <1 x i16> @fcvtzu_v1f64_v1i16(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v1f64_v1i16:
; CHECK: ptrue [[PG:p[0-9]+]].d
; CHECK-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG]]/m, z0.d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %res = fptoui <1 x double> %op1 to <1 x i16>
  ret <1 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i16> @fcvtzu_v2f64_v2i16(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f64_v2i16:
; CHECK: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <2 x double> %op1 to <2 x i16>
  ret <2 x i16> %res
}

define <4 x i16> @fcvtzu_v4f64_v4i16(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvtzu_v4f64_v4i16:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptoui <4 x double> %op1 to <4 x i16>
  ret <4 x i16> %res
}

define <8 x i16> @fcvtzu_v8f64_v8i16(<8 x double>* %a) #0 {
; CHECK-LABEL: fcvtzu_v8f64_v8i16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: fcvtzu [[CVT_HI:z[0-9]+]].d, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: fcvtzu [[CVT_LO:z[0-9]+]].d, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: uzp1 [[UZP_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[UZP_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: uzp1 z0.h, [[UZP_LO]].h, [[UZP_LO]].h
; VBITS_EQ_256-DAG: uzp1 z[[RES_HI:[0-9]+]].h, [[UZP_HI]].h, [[UZP_HI]].h
; VBITS_EQ_256-NEXT: mov v0.d[1], v[[RES_HI]].d[0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptoui <8 x double> %op1 to <8 x i16>
  ret <8 x i16> %res
}

define void @fcvtzu_v16f64_v16i16(<16 x double>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f64_v16i16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptoui <16 x double> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzu_v32f64_v32i16(<32 x double>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f64_v32i16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptoui <32 x double> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

;
; FCVTZU D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x i32> @fcvtzu_v1f64_v1i32(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v1f64_v1i32:
; CHECK: fcvtzu v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <1 x double> %op1 to <1 x i32>
  ret <1 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i32> @fcvtzu_v2f64_v2i32(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f64_v2i32:
; CHECK: fcvtzu v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <2 x double> %op1 to <2 x i32>
  ret <2 x i32> %res
}

define <4 x i32> @fcvtzu_v4f64_v4i32(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvtzu_v4f64_v4i32:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 z0.s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptoui <4 x double> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzu_v8f64_v8i32(<8 x double>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f64_v8i32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].s, vl4
; VBITS_EQ_256-DAG: fcvtzu [[CVT_HI:z[0-9]+]].d, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: fcvtzu [[CVT_LO:z[0-9]+]].d, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].s, [[PG3]], [[RES_LO]].s, [[RES_HI]].s
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[RES]].s }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptoui <8 x double> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzu_v16f64_v16i32(<16 x double>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f64_v16i32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptoui <16 x double> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzu_v32f64_v32i32(<32 x double>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f64_v32i32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvtzu [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptoui <32 x double> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

;
; FCVTZU D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzu_v1f64_v1i64(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v1f64_v1i64:
; CHECK: fcvtzu x8, d0
; CHECK: fmov d0, x8
; CHECK-NEXT: ret
  %res = fptoui <1 x double> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @fcvtzu_v2f64_v2i64(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzu_v2f64_v2i64:
; CHECK: fcvtzu v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptoui <2 x double> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzu_v4f64_v4i64(<4 x double>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v4f64_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptoui <4 x double> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzu_v8f64_v8i64(<8 x double>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v8f64_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: fcvtzu [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[LO]].d
; VBITS_EQ_256-DAG: fcvtzu [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptoui <8 x double> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzu_v16f64_v16i64(<16 x double>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v16f64_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptoui <16 x double> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzu_v32f64_v32i64(<32 x double>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzu_v32f64_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzu [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptoui <32 x double> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}

;
; FCVTZS H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x i16> @fcvtzs_v4f16_v4i16(<4 x half> %op1) #0 {
; CHECK-LABEL: fcvtzs_v4f16_v4i16:
; CHECK: fcvtzs v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = fptosi <4 x half> %op1 to <4 x i16>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define void @fcvtzs_v8f16_v8i16(<8 x half>* %a, <8 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f16_v8i16:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: fcvtzs v0.8h, v0.8h
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptosi <8 x half> %op1 to <8 x i16>
  store <8 x i16> %res, <8 x i16>* %b
  ret void
}

define void @fcvtzs_v16f16_v16i16(<16 x half>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f16_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptosi <16 x half> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzs_v32f16_v32i16(<32 x half>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f16_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[LO]].h
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x1, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptosi <32 x half> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

define void @fcvtzs_v64f16_v64i16(<64 x half>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v64f16_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fptosi <64 x half> %op1 to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %b
  ret void
}

define void @fcvtzs_v128f16_v128i16(<128 x half>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v128f16_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %res = fptosi <128 x half> %op1 to <128 x i16>
  store <128 x i16> %res, <128 x i16>* %b
  ret void
}

;
; FCVTZS H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x i32> @fcvtzs_v2f16_v2i32(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f16_v2i32:
; CHECK: fcvtl v0.4s, v0.4h
; CHECK-NEXT: fcvtzs v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptosi <2 x half> %op1 to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @fcvtzs_v4f16_v4i32(<4 x half> %op1) #0 {
; CHECK-LABEL: fcvtzs_v4f16_v4i32:
; CHECK: fcvtzs v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptosi <4 x half> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzs_v8f16_v8i32(<8 x half>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f16_v8i32:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG]]/m, [[UPK]].h
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptosi <8 x half> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzs_v16f16_v16i32(<16 x half>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f16_v16i32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[VEC:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].s, [[VEC]].h
; VBITS_EQ_256-DAG: ext [[VEC_HI:z[0-9]+]].b, [[VEC]].b, [[VEC]].b, #16
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].s, [[VEC_HI]].h
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].s, [[PG2]]/m, [[UPK_LO]].h
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].s, [[PG2]]/m, [[UPK_HI]].h
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG2]], [x1, x[[NUMELTS]], lsl #2]
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptosi <16 x half> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzs_v32f16_v32i32(<32 x half>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f16_v32i32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptosi <32 x half> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

define void @fcvtzs_v64f16_v64i32(<64 x half>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v64f16_v64i32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fptosi <64 x half> %op1 to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %b
  ret void
}

;
; FCVTZS H -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzs_v1f16_v1i64(<1 x half> %op1) #0 {
; CHECK-LABEL: fcvtzs_v1f16_v1i64:
; CHECK: fcvtzs x8, h0
; CHECK-NEXT: fmov d0, x8
; CHECK-NEXT: ret
  %res = fptosi <1 x half> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x i64> @fcvtzs_v2f16_v2i64(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f16_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvtzs z0.d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: ret
  %res = fptosi <2 x half> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzs_v4f16_v4i64(<4 x half>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v4f16_v4i64:
; CHECK: ldr d[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x half>, <4 x half>* %a
  %res = fptosi <4 x half> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzs_v8f16_v8i64(<8 x half>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f16_v8i64:
; VBITS_GE_512: ldr q[[OP:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ldr q[[OP:[0-9]+]], [x0]
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ext v[[HI:[0-9]+]].16b, v[[LO:[0-9]+]].16b, v[[OP]].16b, #8
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK2_LO]].h
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK2_HI]].h
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG2]], [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fptosi <8 x half> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzs_v16f16_v16i64(<16 x half>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f16_v16i64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fptosi <16 x half> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzs_v32f16_v32i64(<32 x half>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f16_v32i64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK]].s
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fptosi <32 x half> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}

;
; FCVTZS S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x i16> @fcvtzs_v2f32_v2i16(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f32_v2i16:
; CHECK: fcvtzs v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = fptosi <2 x float> %op1 to <2 x i16>
  ret <2 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i16> @fcvtzs_v4f32_v4i16(<4 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v4f32_v4i16:
; CHECK: fcvtzs v1.4s, v0.4s
; CHECK-NEXT: mov w8, v1.s[1]
; CHECK-NEXT: mov w9, v1.s[2]
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: mov v0.h[1], w8
; CHECK-NEXT: mov w8, v1.s[3]
; CHECK-NEXT: mov v0.h[2], w9
; CHECK-NEXT: mov v0.h[3], w8
; CHECK-NEXT: ret
  %res = fptosi <4 x float> %op1 to <4 x i16>
  ret <4 x i16> %res
}

define <8 x i16> @fcvtzs_v8f32_v8i16(<8 x float>* %a) #0 {
; CHECK-LABEL: fcvtzs_v8f32_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: fcvtzs [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; CHECK-NEXT: uzp1 z0.h, [[CVT]].h, [[CVT]].h
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptosi <8 x float> %op1 to <8 x i16>
  ret <8 x i16> %res
}

define void @fcvtzs_v16f32_v16i16(<16 x float>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f32_v16i16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: fcvtzs [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].h, vl8
; VBITS_EQ_256-DAG: fcvtzs [[CVT_HI:z[0-9]+]].s, [[PG2]]/m, [[HI]].s
; VBITS_EQ_256-DAG: fcvtzs [[CVT_LO:z[0-9]+]].s, [[PG2]]/m, [[LO]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].h, [[CVT_LO]].h, [[CVT_LO]].h
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].h, [[CVT_HI]].h, [[CVT_HI]].h
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].h, [[PG3]], [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: st1h { [[RES]].h }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptosi <16 x float> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzs_v32f32_v32i16(<32 x float>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f32_v32i16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: fcvtzs [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptosi <32 x float> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

define void @fcvtzs_v64f32_v64i16(<64 x float>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v64f32_v64i16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptosi <64 x float> %op1 to <64 x i16>
  store <64 x i16> %res, <64 x i16>* %b
  ret void
}

;
; FCVTZS S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x i32> @fcvtzs_v2f32_v2i32(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f32_v2i32:
; CHECK: fcvtzs v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = fptosi <2 x float> %op1 to <2 x i32>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @fcvtzs_v4f32_v4i32(<4 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v4f32_v4i32:
; CHECK: fcvtzs v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = fptosi <4 x float> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzs_v8f32_v8i32(<8 x float>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f32_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptosi <8 x float> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzs_v16f32_v16i32(<16 x float>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f32_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[LO]].s
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x1, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptosi <16 x float> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzs_v32f32_v32i32(<32 x float>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f32_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptosi <32 x float> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

define void @fcvtzs_v64f32_v64i32(<64 x float>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v64f32_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptosi <64 x float> %op1 to <64 x i32>
  store <64 x i32> %res, <64 x i32>* %b
  ret void
}

;
; FCVTZS S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzs_v1f32_v1i64(<1 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v1f32_v1i64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <1 x float> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @fcvtzs_v2f32_v2i64(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f32_v2i64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <2 x float> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzs_v4f32_v4i64(<4 x float>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v4f32_v4i64:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[OP]].s
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK]].s
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x float>, <4 x float>* %a
  %res = fptosi <4 x float> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzs_v8f32_v8i64(<8 x float>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f32_v8i64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG1]]/m, [[UPK]].s
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VEC:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].d, [[VEC]].s
; VBITS_EQ_256-DAG: ext [[VEC_HI:z[0-9]+]].b, [[VEC]].b, [[VEC]].b, #16
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].d, [[VEC]].s
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK_LO]].s
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK_HI]].s
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG2]], [x1, x[[NUMELTS]], lsl #3]
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptosi <8 x float> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzs_v16f32_v16i64(<16 x float>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f32_v16i64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptosi <16 x float> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzs_v32f32_v32i64(<32 x float>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f32_v32i64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptosi <32 x float> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}


;
; FCVTZS D -> H
;

; v1f64 is perfered to be widened to v4f64, so use SVE
define <1 x i16> @fcvtzs_v1f64_v1i16(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v1f64_v1i16:
; CHECK: ptrue [[PG:p[0-9]+]].d
; CHECK-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG]]/m, z0.d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %res = fptosi <1 x double> %op1 to <1 x i16>
  ret <1 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i16> @fcvtzs_v2f64_v2i16(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f64_v2i16:
; CHECK: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <2 x double> %op1 to <2 x i16>
  ret <2 x i16> %res
}

define <4 x i16> @fcvtzs_v4f64_v4i16(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvtzs_v4f64_v4i16:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptosi <4 x double> %op1 to <4 x i16>
  ret <4 x i16> %res
}

define <8 x i16> @fcvtzs_v8f64_v8i16(<8 x double>* %a) #0 {
; CHECK-LABEL: fcvtzs_v8f64_v8i16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: fcvtzs [[CVT_HI:z[0-9]+]].d, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: fcvtzs [[CVT_LO:z[0-9]+]].d, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: uzp1 [[UZP_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[UZP_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: uzp1 z0.h, [[UZP_LO]].h, [[UZP_LO]].h
; VBITS_EQ_256-DAG: uzp1 z[[RES_HI:[0-9]+]].h, [[UZP_HI]].h, [[UZP_HI]].h
; VBITS_EQ_256-NEXT: mov v0.d[1], v[[RES_HI]].d[0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptosi <8 x double> %op1 to <8 x i16>
  ret <8 x i16> %res
}

define void @fcvtzs_v16f64_v16i16(<16 x double>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f64_v16i16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptosi <16 x double> %op1 to <16 x i16>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @fcvtzs_v32f64_v32i16(<32 x double>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f64_v32i16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptosi <32 x double> %op1 to <32 x i16>
  store <32 x i16> %res, <32 x i16>* %b
  ret void
}

;
; FCVTZS D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x i32> @fcvtzs_v1f64_v1i32(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v1f64_v1i32:
; CHECK: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <1 x double> %op1 to <1 x i32>
  ret <1 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i32> @fcvtzs_v2f64_v2i32(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f64_v2i32:
; CHECK: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: xtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <2 x double> %op1 to <2 x i32>
  ret <2 x i32> %res
}

define <4 x i32> @fcvtzs_v4f64_v4i32(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvtzs_v4f64_v4i32:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 z0.s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptosi <4 x double> %op1 to <4 x i32>
  ret <4 x i32> %res
}

define void @fcvtzs_v8f64_v8i32(<8 x double>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f64_v8i32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].s, vl4
; VBITS_EQ_256-DAG: fcvtzs [[CVT_HI:z[0-9]+]].d, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: fcvtzs [[CVT_LO:z[0-9]+]].d, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].s, [[PG3]], [[RES_LO]].s, [[RES_HI]].s
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[RES]].s }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptosi <8 x double> %op1 to <8 x i32>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @fcvtzs_v16f64_v16i32(<16 x double>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f64_v16i32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptosi <16 x double> %op1 to <16 x i32>
  store <16 x i32> %res, <16 x i32>* %b
  ret void
}

define void @fcvtzs_v32f64_v32i32(<32 x double>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f64_v32i32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvtzs [[CVT:z[0-9]+]].d, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptosi <32 x double> %op1 to <32 x i32>
  store <32 x i32> %res, <32 x i32>* %b
  ret void
}

;
; FCVTZS D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x i64> @fcvtzs_v1f64_v1i64(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v1f64_v1i64:
; CHECK: fcvtzs x8, d0
; CHECK: fmov d0, x8
; CHECK-NEXT: ret
  %res = fptosi <1 x double> %op1 to <1 x i64>
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @fcvtzs_v2f64_v2i64(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvtzs_v2f64_v2i64:
; CHECK: fcvtzs v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = fptosi <2 x double> %op1 to <2 x i64>
  ret <2 x i64> %res
}

define void @fcvtzs_v4f64_v4i64(<4 x double>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v4f64_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptosi <4 x double> %op1 to <4 x i64>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

define void @fcvtzs_v8f64_v8i64(<8 x double>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v8f64_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: fcvtzs [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[LO]].d
; VBITS_EQ_256-DAG: fcvtzs [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x1, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptosi <8 x double> %op1 to <8 x i64>
  store <8 x i64> %res, <8 x i64>* %b
  ret void
}

define void @fcvtzs_v16f64_v16i64(<16 x double>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v16f64_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptosi <16 x double> %op1 to <16 x i64>
  store <16 x i64> %res, <16 x i64>* %b
  ret void
}

define void @fcvtzs_v32f64_v32i64(<32 x double>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: fcvtzs_v32f64_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: fcvtzs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptosi <32 x double> %op1 to <32 x i64>
  store <32 x i64> %res, <32 x i64>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
