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
; UCVTF H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @ucvtf_v4i16_v4f16(<4 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f16:
; CHECK: ucvtf v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = uitofp <4 x i16> %op1 to <4 x half>
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define void @ucvtf_v8i16_v8f16(<8 x i16>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i16_v8f16:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: ucvtf v0.8h, v0.8h
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @ucvtf_v16i16_v16f16(<16 x i16>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i16_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i16_v32f16(<32 x i16>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i16_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[LO]].h
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[HI]].h
; VBITS_EQ_256-NEXT: st1h { [[RES_LO]].h }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1h { [[RES_HI]].h }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @ucvtf_v64i16_v64f16(<64 x i16>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v64i16_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = uitofp <64 x i16> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

define void @ucvtf_v128i16_v128f16(<128 x i16>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v128i16_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = uitofp <128 x i16> %op1 to <128 x half>
  store <128 x half> %res, <128 x half>* %b
  ret void
}

;
; UCVTF H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @ucvtf_v2i16_v2f32(<2 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i16_v2f32:
; CHECK: movi d1, #0x00ffff0000ffff
; CHECK-NEXT: and v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ucvtf v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = uitofp <2 x i16> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @ucvtf_v4i16_v4f32(<4 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f32:
; CHECK: ucvtf v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = uitofp <4 x i16> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i16_v8f32(<8 x i16>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i16_v8f32:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[UPK]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i16_v16f32(<16 x i16>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i16_v16f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[VEC:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1h { [[VEC:z[0-9]+]].h }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: uunpklo [[UPK_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[UPK_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].s, [[PG2]]/m, [[UPK_LO]].s
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].s, [[PG2]]/m, [[UPK_HI]].s
; VBITS_EQ_256-NEXT: st1w { [[RES_HI]].s }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: st1w { [[RES_LO]].s }, [[PG2]], [x1]
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i16_v32f32(<32 x i16>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i16_v32f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @ucvtf_v64i16_v64f32(<64 x i16>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v64i16_v64f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = uitofp <64 x i16> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; UCVTF H -> D
;

; v1i16 is perfered to be widened to v4i16, which pushes the output into SVE types, so use SVE
define <1 x double> @ucvtf_v1i16_v1f64(<1 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i16_v1f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: ucvtf z0.d, [[PG]]/m, [[UPK2]].d
; CHECK-NEXT: ret
  %res = uitofp <1 x i16> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i16_v2f64(<2 x i16> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i16_v2f64:
; CHECK: movi d1, #0x00ffff0000ffff
; CHECK-NEXT: and v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ushll v0.2d, v0.2s, #0
; CHECK-NEXT: ucvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <2 x i16> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i16_v4f64(<4 x i16>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i16_v4f64:
; CHECK: ldr d[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i16>, <4 x i16>* %a
  %res = uitofp <4 x i16> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i16_v8f64(<8 x i16>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i16_v8f64:
; VBITS_GE_512: ldr q[[OP:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ldr q[[OP:[0-9]+]], [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: ext v[[HI:[0-9]+]].16b, v[[LO:[0-9]+]].16b, v[[OP]].16b, #8
; VBITS_EQ_256-NEXT: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-NEXT: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-NEXT: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK2_LO]].d
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK2_HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = uitofp <8 x i16> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i16_v16f64(<16 x i16>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i16_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = uitofp <16 x i16> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i16_v32f64(<32 x i16>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i16_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK]].s
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = uitofp <32 x i16> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; UCVTF S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x half> @ucvtf_v2i32_v2f16(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f16:
; CHECK: ucvtf v0.4s, v0.4s
; CHECK-NEXT: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = uitofp <2 x i32> %op1 to <2 x half>
  ret <2 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @ucvtf_v4i32_v4f16(<4 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f16:
; CHECK: ucvtf v0.4s, v0.4s
; CHECK-NEXT: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = uitofp <4 x i32> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @ucvtf_v8i32_v8f16(<8 x i32>* %a) #0 {
; CHECK-LABEL: ucvtf_v8i32_v8f16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; CHECK-NEXT: uzp1 z0.h, [[CVT]].h, [[CVT]].h
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @ucvtf_v16i32_v16f16(<16 x i32>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i32_v16f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_EQ_256-NEXT: ptrue [[PG3:p[0-9]+]].h, vl8
; VBITS_EQ_256-NEXT: ucvtf [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].s
; VBITS_EQ_256-NEXT: ucvtf [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES_LO:z[0-9]+]].h, [[CVT_LO]].h, [[CVT_LO]].h
; VBITS_EQ_256-NEXT: uzp1 [[RES_HI:z[0-9]+]].h, [[CVT_HI]].h, [[CVT_HI]].h
; VBITS_EQ_256-NEXT: splice [[RES:z[0-9]+]].h, [[PG3]], [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_256-NEXT: ptrue [[PG4:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: st1h { [[RES]].h }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i32_v32f16(<32 x i32>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i32_v32f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @ucvtf_v64i32_v64f16(<64 x i32>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v64i32_v64f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].h, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = uitofp <64 x i32> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; UCVTF S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @ucvtf_v2i32_v2f32(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f32:
; CHECK: ucvtf v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = uitofp <2 x i32> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @ucvtf_v4i32_v4f32(<4 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f32:
; CHECK: ucvtf v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = uitofp <4 x i32> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i32_v8f32(<8 x i32>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i32_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i32_v16f32(<16 x i32>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i32_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[LO]].s
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[HI]].s
; VBITS_EQ_256-NEXT: st1w { [[RES_LO]].s }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1w { [[RES_HI]].s }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i32_v32f32(<32 x i32>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i32_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @ucvtf_v64i32_v64f32(<64 x i32>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v64i32_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = uitofp <64 x i32> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; UCVTF S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @ucvtf_v1i32_v1f64(<1 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i32_v1f64:
; CHECK: ushll v0.2d, v0.2s, #0
; CHECK-NEXT: ucvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <1 x i32> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i32_v2f64(<2 x i32> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i32_v2f64:
; CHECK: ushll v0.2d, v0.2s, #0
; CHECK-NEXT: ucvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <2 x i32> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i32_v4f64(<4 x i32>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i32_v4f64:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[OP]].s
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %res = uitofp <4 x i32> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i32_v8f64(<8 x i32>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i32_v8f64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG1]]/m, [[UPK]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VEC:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1w { [[VEC:z[0-9]+]].s }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: uunpklo [[UPK_LO:z[0-9]+]].d, z[[LO]].s
; VBITS_EQ_256-NEXT: uunpklo [[UPK_HI:z[0-9]+]].d, z[[HI]].s
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK_LO]].d
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK_HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG2]], [x1]
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = uitofp <8 x i32> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i32_v16f64(<16 x i32>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i32_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = uitofp <16 x i32> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i32_v32f64(<32 x i32>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i32_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = uitofp <32 x i32> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}


;
; UCVTF D -> H
;

; Don't use SVE for 64-bit vectors.
define <1 x half> @ucvtf_v1i64_v1f16(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f16:
; CHECK: fmov x8, d0
; CHECK-NEXT: ucvtf h0, x8
; CHECK-NEXT: ret
  %res = uitofp <1 x i64> %op1 to <1 x half>
  ret <1 x half> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x half> @ucvtf_v2i64_v2f16(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f16:
; CHECK: ptrue [[PG:p[0-9]+]].d
; CHECK-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG]]/m, z0.d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %res = uitofp <2 x i64> %op1 to <2 x half>
  ret <2 x half> %res
}

define <4 x half> @ucvtf_v4i64_v4f16(<4 x i64>* %a) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f16:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @ucvtf_v8i64_v8f16(<8 x i64>* %a) #0 {
; CHECK-LABEL: ucvtf_v8i64_v8f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-NEXT: ucvtf [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: ucvtf [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: uzp1 [[UZP_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-NEXT: uzp1 z[[RES_LO:[0-9]+]].h, [[UZP_LO]].h, [[UZP_LO]].h
; VBITS_EQ_256-NEXT: uzp1 z[[RES_HI:[0-9]+]].h, [[UZP_HI]].h, [[UZP_HI]].h
; VBITS_EQ_256-NEXT: mov v[[RES_LO]].d[1], v[[RES_HI]].d[0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @ucvtf_v16i64_v16f16(<16 x i64>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i64_v16f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @ucvtf_v32i64_v32f16(<32 x i64>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i64_v32f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: ucvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; UCVTF D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x float> @ucvtf_v1i64_v1f32(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f32:
; CHECK: ucvtf v0.2d, v0.2d
; CHECK-NEXT: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <1 x i64> %op1 to <1 x float>
  ret <1 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @ucvtf_v2i64_v2f32(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f32:
; CHECK: ucvtf v0.2d, v0.2d
; CHECK-NEXT: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <2 x i64> %op1 to <2 x float>
  ret <2 x float> %res
}

define <4 x float> @ucvtf_v4i64_v4f32(<4 x i64>* %a) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f32:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: ucvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 z0.s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @ucvtf_v8i64_v8f32(<8 x i64>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i64_v8f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: ucvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-NEXT: ptrue [[PG3:p[0-9]+]].s, vl4
; VBITS_EQ_256-NEXT: ucvtf [[CVT_HI:z[0-9]+]].s, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: ucvtf [[CVT_LO:z[0-9]+]].s, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: uzp1 [[RES_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-NEXT: splice [[RES:z[0-9]+]].s, [[PG3]], [[RES_LO]].s, [[RES_HI]].s
; VBITS_EQ_256-NEXT: ptrue [[PG4:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[RES]].s }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @ucvtf_v16i64_v16f32(<16 x i64>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i64_v16f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: ucvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @ucvtf_v32i64_v32f32(<32 x i64>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i64_v32f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: ucvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

;
; UCVTF D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @ucvtf_v1i64_v1f64(<1 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v1i64_v1f64:
; CHECK: fmov x8, d0
; CHECK-NEXT: ucvtf d0, x8
; CHECK-NEXT: ret
  %res = uitofp <1 x i64> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @ucvtf_v2i64_v2f64(<2 x i64> %op1) #0 {
; CHECK-LABEL: ucvtf_v2i64_v2f64:
; CHECK: ucvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = uitofp <2 x i64> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @ucvtf_v4i64_v4f64(<4 x i64>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v4i64_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = uitofp <4 x i64> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @ucvtf_v8i64_v8f64(<8 x i64>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v8i64_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: ucvtf [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: ucvtf [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = uitofp <8 x i64> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @ucvtf_v16i64_v16f64(<16 x i64>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v16i64_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = uitofp <16 x i64> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @ucvtf_v32i64_v32f64(<32 x i64>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: ucvtf_v32i64_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ucvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = uitofp <32 x i64> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; SCVTF H -> H
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @scvtf_v4i16_v4f16(<4 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f16:
; CHECK: scvtf v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = sitofp <4 x i16> %op1 to <4 x half>
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define void @scvtf_v8i16_v8f16(<8 x i16>* %a, <8 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v8i16_v8f16:
; CHECK: ldr q0, [x0]
; CHECK-NEXT: scvtf v0.8h, v0.8h
; CHECK-NEXT: str q0, [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x half>
  store <8 x half> %res, <8 x half>* %b
  ret void
}

define void @scvtf_v16i16_v16f16(<16 x i16>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v16i16_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i16_v32f16(<32 x i16>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v32i16_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1h { [[LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[HI:z[0-9]+]].h }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[LO]].h
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[HI]].h
; VBITS_EQ_256-NEXT: st1h { [[RES_LO]].h }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1h { [[RES_HI]].h }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @scvtf_v64i16_v64f16(<64 x i16>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v64i16_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = sitofp <64 x i16> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

define void @scvtf_v128i16_v128f16(<128 x i16>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v128i16_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = sitofp <128 x i16> %op1 to <128 x half>
  store <128 x half> %res, <128 x half>* %b
  ret void
}

;
; SCVTF H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @scvtf_v2i16_v2f32(<2 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v2i16_v2f32:
; CHECK: shl v0.2s, v0.2s, #16
; CHECK-NEXT: sshr v0.2s, v0.2s, #16
; CHECK-NEXT: scvtf v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = sitofp <2 x i16> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @scvtf_v4i16_v4f32(<4 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f32:
; CHECK: scvtf v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = sitofp <4 x i16> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i16_v8f32(<8 x i16>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v8i16_v8f32:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[UPK:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[UPK]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i16_v16f32(<16 x i16>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v16i16_v16f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[VEC:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1h { [[VEC:z[0-9]+]].h }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: sunpklo [[UPK_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[UPK_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].s, [[PG2]]/m, [[UPK_LO]].s
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].s, [[PG2]]/m, [[UPK_HI]].s
; VBITS_EQ_256-NEXT: st1w { [[RES_HI]].s }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: st1w { [[RES_LO]].s }, [[PG2]], [x1]
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i16_v32f32(<32 x i16>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v32i16_v32f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: sunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @scvtf_v64i16_v64f32(<64 x i16>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v64i16_v64f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: sunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = sitofp <64 x i16> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; SCVTF H -> D
;

; v1i16 is perfered to be widened to v4i16, which pushes the output into SVE types, so use SVE
define <1 x double> @scvtf_v1i16_v1f64(<1 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v1i16_v1f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: sunpklo [[UPK1:z[0-9]+]].s, z0.h
; CHECK-NEXT: sunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: scvtf z0.d, [[PG]]/m, [[UPK2]].d
; CHECK-NEXT: ret
  %res = sitofp <1 x i16> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i16_v2f64(<2 x i16> %op1) #0 {
; CHECK-LABEL: scvtf_v2i16_v2f64:
; CHECK: shl v0.2s, v0.2s, #16
; CHECK-NEXT: sshr v0.2s, v0.2s, #16
; CHECK-NEXT: sshll v0.2d, v0.2s, #0
; CHECK-NEXT: scvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <2 x i16> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i16_v4f64(<4 x i16>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i16_v4f64:
; CHECK: ldr d[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: sunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: sunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i16>, <4 x i16>* %a
  %res = sitofp <4 x i16> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i16_v8f64(<8 x i16>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v8i16_v8f64:
; VBITS_GE_512: ldr q[[OP:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; VBITS_GE_512-NEXT: sunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ldr q[[OP:[0-9]+]], [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: ext v[[HI:[0-9]+]].16b, v[[LO:[0-9]+]].16b, v[[OP]].16b, #8
; VBITS_EQ_256-NEXT: sunpklo [[UPK1_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[UPK1_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-NEXT: sunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-NEXT: sunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK1_HI]].s
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK2_LO]].d
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK2_HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = sitofp <8 x i16> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i16_v16f64(<16 x i16>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v16i16_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: sunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: sunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = sitofp <16 x i16> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i16_v32f64(<32 x i16>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v32i16_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: sunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: sunpklo [[UPK2:z[0-9]+]].d, [[UPK]].s
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = sitofp <32 x i16> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; SCVTF S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x half> @scvtf_v2i32_v2f16(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f16:
; CHECK: scvtf v0.4s, v0.4s
; CHECK-NEXT: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = sitofp <2 x i32> %op1 to <2 x half>
  ret <2 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @scvtf_v4i32_v4f16(<4 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f16:
; CHECK: scvtf v0.4s, v0.4s
; CHECK-NEXT: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = sitofp <4 x i32> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @scvtf_v8i32_v8f16(<8 x i32>* %a) #0 {
; CHECK-LABEL: scvtf_v8i32_v8f16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; CHECK-NEXT: uzp1 z0.h, [[CVT]].h, [[CVT]].h
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @scvtf_v16i32_v16f16(<16 x i32>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v16i32_v16f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_EQ_256-NEXT: ptrue [[PG3:p[0-9]+]].h, vl8
; VBITS_EQ_256-NEXT: scvtf [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].s
; VBITS_EQ_256-NEXT: scvtf [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES_LO:z[0-9]+]].h, [[CVT_LO]].h, [[CVT_LO]].h
; VBITS_EQ_256-NEXT: uzp1 [[RES_HI:z[0-9]+]].h, [[CVT_HI]].h, [[CVT_HI]].h
; VBITS_EQ_256-NEXT: splice [[RES:z[0-9]+]].h, [[PG3]], [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_256-NEXT: ptrue [[PG4:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: st1h { [[RES]].h }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i32_v32f16(<32 x i32>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v32i32_v32f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @scvtf_v64i32_v64f16(<64 x i32>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v64i32_v64f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].h, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = sitofp <64 x i32> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; SCVTF S -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @scvtf_v2i32_v2f32(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f32:
; CHECK: scvtf v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = sitofp <2 x i32> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @scvtf_v4i32_v4f32(<4 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f32:
; CHECK: scvtf v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = sitofp <4 x i32> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i32_v8f32(<8 x i32>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v8i32_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i32_v16f32(<16 x i32>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v16i32_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1w { [[LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1w { [[HI:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[LO]].s
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[HI]].s
; VBITS_EQ_256-NEXT: st1w { [[RES_LO]].s }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1w { [[RES_HI]].s }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i32_v32f32(<32 x i32>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v32i32_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @scvtf_v64i32_v64f32(<64 x i32>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v64i32_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = sitofp <64 x i32> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; SCVTF S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @scvtf_v1i32_v1f64(<1 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v1i32_v1f64:
; CHECK: sshll v0.2d, v0.2s, #0
; CHECK-NEXT: scvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <1 x i32> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i32_v2f64(<2 x i32> %op1) #0 {
; CHECK-LABEL: scvtf_v2i32_v2f64:
; CHECK: sshll v0.2d, v0.2s, #0
; CHECK-NEXT: scvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <2 x i32> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i32_v4f64(<4 x i32>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i32_v4f64:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: sunpklo [[UPK:z[0-9]+]].d, z[[OP]].s
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %res = sitofp <4 x i32> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i32_v8f64(<8 x i32>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v8i32_v8f64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG1]]/m, [[UPK]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VEC:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1w { [[VEC:z[0-9]+]].s }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: sunpklo [[UPK_LO:z[0-9]+]].d, z[[LO]].s
; VBITS_EQ_256-NEXT: sunpklo [[UPK_HI:z[0-9]+]].d, z[[HI]].s
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK_LO]].d
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK_HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG2]], [x8]
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG2]], [x1]
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = sitofp <8 x i32> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i32_v16f64(<16 x i32>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v16i32_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: sunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = sitofp <16 x i32> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i32_v32f64(<32 x i32>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v32i32_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: sunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = sitofp <32 x i32> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}


;
; SCVTF D -> H
;

; Don't use SVE for 64-bit vectors.
define <1 x half> @scvtf_v1i64_v1f16(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f16:
; CHECK: fmov x8, d0
; CHECK-NEXT: scvtf h0, x8
; CHECK-NEXT: ret
  %res = sitofp <1 x i64> %op1 to <1 x half>
  ret <1 x half> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x half> @scvtf_v2i64_v2f16(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f16:
; CHECK: ptrue [[PG:p[0-9]+]].d
; CHECK-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG]]/m, z0.d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %res = sitofp <2 x i64> %op1 to <2 x half>
  ret <2 x half> %res
}

define <4 x half> @scvtf_v4i64_v4f16(<4 x i64>* %a) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f16:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @scvtf_v8i64_v8f16(<8 x i64>* %a) #0 {
; CHECK-LABEL: scvtf_v8i64_v8f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-NEXT: scvtf [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: scvtf [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: uzp1 [[UZP_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-NEXT: uzp1 z[[RES_LO:[0-9]+]].h, [[UZP_LO]].h, [[UZP_LO]].h
; VBITS_EQ_256-NEXT: uzp1 z[[RES_HI:[0-9]+]].h, [[UZP_HI]].h, [[UZP_HI]].h
; VBITS_EQ_256-NEXT: mov v[[RES_LO]].d[1], v[[RES_HI]].d[0]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @scvtf_v16i64_v16f16(<16 x i64>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v16i64_v16f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @scvtf_v32i64_v32f16(<32 x i64>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: scvtf_v32i64_v32f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: scvtf [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; SCVTF D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x float> @scvtf_v1i64_v1f32(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f32:
; CHECK: scvtf v0.2d, v0.2d
; CHECK-NEXT: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <1 x i64> %op1 to <1 x float>
  ret <1 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @scvtf_v2i64_v2f32(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f32:
; CHECK: scvtf v0.2d, v0.2d
; CHECK-NEXT: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <2 x i64> %op1 to <2 x float>
  ret <2 x float> %res
}

define <4 x float> @scvtf_v4i64_v4f32(<4 x i64>* %a) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f32:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: scvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 z0.s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @scvtf_v8i64_v8f32(<8 x i64>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v8i64_v8f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: scvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: add x8, x0, #32
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-NEXT: ptrue [[PG3:p[0-9]+]].s, vl4
; VBITS_EQ_256-NEXT: scvtf [[CVT_HI:z[0-9]+]].s, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: scvtf [[CVT_LO:z[0-9]+]].s, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: uzp1 [[RES_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-NEXT: splice [[RES:z[0-9]+]].s, [[PG3]], [[RES_LO]].s, [[RES_HI]].s
; VBITS_EQ_256-NEXT: ptrue [[PG4:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[RES]].s }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @scvtf_v16i64_v16f32(<16 x i64>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v16i64_v16f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: scvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @scvtf_v32i64_v32f32(<32 x i64>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: scvtf_v32i64_v32f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: scvtf [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

;
; SCVTF D -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @scvtf_v1i64_v1f64(<1 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v1i64_v1f64:
; CHECK: fmov x8, d0
; CHECK-NEXT: scvtf d0, x8
; CHECK-NEXT: ret
  %res = sitofp <1 x i64> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @scvtf_v2i64_v2f64(<2 x i64> %op1) #0 {
; CHECK-LABEL: scvtf_v2i64_v2f64:
; CHECK: scvtf v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = sitofp <2 x i64> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @scvtf_v4i64_v4f64(<4 x i64>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v4i64_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = sitofp <4 x i64> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @scvtf_v8i64_v8f64(<8 x i64>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v8i64_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-NEXT: add x8, x0, #32
; VBITS_EQ_256-NEXT: ld1d { [[LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1d { [[HI:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-NEXT: add x8, x1, #32
; VBITS_EQ_256-NEXT: scvtf [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[LO]].d
; VBITS_EQ_256-NEXT: scvtf [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[HI]].d
; VBITS_EQ_256-NEXT: st1d { [[RES_LO]].d }, [[PG]], [x1]
; VBITS_EQ_256-NEXT: st1d { [[RES_HI]].d }, [[PG]], [x8]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = sitofp <8 x i64> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @scvtf_v16i64_v16f64(<16 x i64>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v16i64_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = sitofp <16 x i64> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @scvtf_v32i64_v32f64(<32 x i64>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: scvtf_v32i64_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: scvtf [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = sitofp <32 x i64> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
