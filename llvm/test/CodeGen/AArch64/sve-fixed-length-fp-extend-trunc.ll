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
; FCVT H -> S
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @fcvt_v2f16_v2f32(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvt_v2f16_v2f32:
; CHECK: fcvtl v0.4s, v0.4h
; CHECK-NEXT: ret
  %res = fpext <2 x half> %op1 to <2 x float>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fcvt_v4f16_v4f32(<4 x half> %op1) #0 {
; CHECK-LABEL: fcvt_v4f16_v4f32:
; CHECK: fcvtl v0.4s, v0.4h
; CHECK-NEXT: ret
  %res = fpext <4 x half> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @fcvt_v8f16_v8f32(<8 x half>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v8f16_v8f32:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: fcvt [[RES:z[0-9]+]].s, [[PG]]/m, [[UPK]].h
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fpext <8 x half> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @fcvt_v16f16_v16f32(<16 x half>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v16f16_v16f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_512-NEXT: fcvt [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[VEC:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1h { [[VEC:z[0-9]+]].h }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].s, z[[LO]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-DAG: fcvt [[RES_LO:z[0-9]+]].s, [[PG2]]/m, [[UPK_LO]].h
; VBITS_EQ_256-DAG: fcvt [[RES_HI:z[0-9]+]].s, [[PG2]]/m, [[UPK_HI]].h
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG2]], [x1, x[[NUMELTS]], lsl #2]
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fpext <16 x half> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @fcvt_v32f16_v32f32(<32 x half>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v32f16_v32f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: fcvt [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fpext <32 x half> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

define void @fcvt_v64f16_v64f32(<64 x half>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v64f16_v64f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: fcvt [[RES:z[0-9]+]].s, [[PG2]]/m, [[UPK]].h
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %res = fpext <64 x half> %op1 to <64 x float>
  store <64 x float> %res, <64 x float>* %b
  ret void
}

;
; FCVT H -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @fcvt_v1f16_v1f64(<1 x half> %op1) #0 {
; CHECK-LABEL: fcvt_v1f16_v1f64:
; CHECK: fcvt d0, h0
; CHECK-NEXT: ret
  %res = fpext <1 x half> %op1 to <1 x double>
  ret <1 x double> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x double> @fcvt_v2f16_v2f64(<2 x half> %op1) #0 {
; CHECK-LABEL: fcvt_v2f16_v2f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvt z0.d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: ret
  %res = fpext <2 x half> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @fcvt_v4f16_v4f64(<4 x half>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v4f16_v4f64:
; CHECK: ldr d[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; CHECK-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; CHECK-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x half>, <4 x half>* %a
  %res = fpext <4 x half> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @fcvt_v8f16_v8f64(<8 x half>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v8f16_v8f64:
; VBITS_GE_512: ldr q[[OP:[0-9]+]], [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK1:z[0-9]+]].s, z[[OP]].h
; VBITS_GE_512-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_512-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK2]].h
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ldr q[[OP:[0-9]+]], [x0]
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ext v[[HI:[0-9]+]].16b, v[[OP]].16b, v[[OP]].16b, #8
; VBITS_EQ_256-DAG: uunpklo [[UPK1_LO:z[0-9]+]].s, z[[OP]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK1_HI:z[0-9]+]].s, z[[HI]].h
; VBITS_EQ_256-DAG: uunpklo [[UPK2_LO:z[0-9]+]].d, [[UPK1_LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK2_HI:z[0-9]+]].d, [[UPK2_HI]].s
; VBITS_EQ_256-DAG: fcvt [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[UPK2_LO]].h
; VBITS_EQ_256-DAG: fcvt [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[UPK2_HI]].h
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x1, x[[NUMELTS]], lsl #3]
  %op1 = load <8 x half>, <8 x half>* %a
  %res = fpext <8 x half> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @fcvt_v16f16_v16f64(<16 x half>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v16f16_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_1024-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK1]].s
; VBITS_GE_1024-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %res = fpext <16 x half> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @fcvt_v32f16_v32f64(<32 x half>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v32f16_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: ld1h { [[OP:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK1:z[0-9]+]].s, [[OP]].h
; VBITS_GE_2048-NEXT: uunpklo [[UPK2:z[0-9]+]].d, [[UPK]].s
; VBITS_GE_2048-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK2]].h
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %res = fpext <32 x half> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; FCVT S -> D
;

; Don't use SVE for 64-bit vectors.
define <1 x double> @fcvt_v1f32_v1f64(<1 x float> %op1) #0 {
; CHECK-LABEL: fcvt_v1f32_v1f64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: ret
  %res = fpext <1 x float> %op1 to <1 x double>
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fcvt_v2f32_v2f64(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvt_v2f32_v2f64:
; CHECK: fcvtl v0.2d, v0.2s
; CHECK-NEXT: ret
  %res = fpext <2 x float> %op1 to <2 x double>
  ret <2 x double> %res
}

define void @fcvt_v4f32_v4f64(<4 x float>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v4f32_v4f64:
; CHECK: ldr q[[OP:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[UPK:z[0-9]+]].d, z[[OP]].s
; CHECK-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG]]/m, [[UPK]].s
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x float>, <4 x float>* %a
  %res = fpext <4 x float> %op1 to <4 x double>
  store <4 x double> %res, <4 x double>* %b
  ret void
}

define void @fcvt_v8f32_v8f64(<8 x float>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v8f32_v8f64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_512-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG1]]/m, [[UPK]].s
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation - fixed type extract_subvector codegen is poor currently.
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[VEC:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: st1w { [[VEC:z[0-9]+]].s }, [[PG1]], [x8]
; VBITS_EQ_256-DAG: ldp q[[LO:[0-9]+]], q[[HI:[0-9]+]], [sp]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: uunpklo [[UPK_LO:z[0-9]+]].d, z[[LO]].s
; VBITS_EQ_256-DAG: uunpklo [[UPK_HI:z[0-9]+]].d, z[[HI]].s
; VBITS_EQ_256-DAG: fcvt [[RES_LO:z[0-9]+]].d, [[PG2]]/m, [[UPK_LO]].s
; VBITS_EQ_256-DAG: fcvt [[RES_HI:z[0-9]+]].d, [[PG2]]/m, [[UPK_HI]].s
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG2]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG2]], [x1, x[[NUMELTS]], lsl #3]
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fpext <8 x float> %op1 to <8 x double>
  store <8 x double> %res, <8 x double>* %b
  ret void
}

define void @fcvt_v16f32_v16f64(<16 x float>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v16f32_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_1024-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fpext <16 x float> %op1 to <16 x double>
  store <16 x double> %res, <16 x double>* %b
  ret void
}

define void @fcvt_v32f32_v32f64(<32 x float>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fcvt_v32f32_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: uunpklo [[UPK:z[0-9]+]].d, [[OP]].s
; VBITS_GE_2048-NEXT: fcvt [[RES:z[0-9]+]].d, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fpext <32 x float> %op1 to <32 x double>
  store <32 x double> %res, <32 x double>* %b
  ret void
}

;
; FCVT S -> H
;

; Don't use SVE for 64-bit vectors.
define <2 x half> @fcvt_v2f32_v2f16(<2 x float> %op1) #0 {
; CHECK-LABEL: fcvt_v2f32_v2f16:
; CHECK: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = fptrunc <2 x float> %op1 to <2 x half>
  ret <2 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x half> @fcvt_v4f32_v4f16(<4 x float> %op1) #0 {
; CHECK-LABEL: fcvt_v4f32_v4f16:
; CHECK: fcvtn v0.4h, v0.4s
; CHECK-NEXT: ret
  %res = fptrunc <4 x float> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @fcvt_v8f32_v8f16(<8 x float>* %a) #0 {
; CHECK-LABEL: fcvt_v8f32_v8f16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; CHECK-NEXT: uzp1 z0.h, [[CVT]].h, [[CVT]].h
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %res = fptrunc <8 x float> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @fcvt_v16f32_v16f16(<16 x float>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v16f32_v16f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[LO:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[HI:z[0-9]+]].s }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].h, vl8
; VBITS_EQ_256-DAG: fcvt [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].s
; VBITS_EQ_256-DAG: fcvt [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].h, [[CVT_LO]].h, [[CVT_LO]].h
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].h, [[CVT_HI]].h, [[CVT_HI]].h
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].h, [[PG3]], [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: st1h { [[RES]].h }, [[PG4]], [x1]
  %op1 = load <16 x float>, <16 x float>* %a
  %res = fptrunc <16 x float> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @fcvt_v32f32_v32f16(<32 x float>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v32f32_v32f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %res = fptrunc <32 x float> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

define void @fcvt_v64f32_v64f16(<64 x float>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v64f32_v64f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: fcvt [[RES:z[0-9]+]].h, [[PG2]]/m, [[UPK]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[CVT]].h, [[CVT]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %res = fptrunc <64 x float> %op1 to <64 x half>
  store <64 x half> %res, <64 x half>* %b
  ret void
}

;
; FCVT D -> H
;

; Don't use SVE for 64-bit vectors.
define <1 x half> @fcvt_v1f64_v1f16(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvt_v1f64_v1f16:
; CHECK: fcvt h0, d0
; CHECK-NEXT: ret
  %res = fptrunc <1 x double> %op1 to <1 x half>
  ret <1 x half> %res
}

; v2f16 is not legal for NEON, so use SVE
define <2 x half> @fcvt_v2f64_v2f16(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvt_v2f64_v2f16:
; CHECK: ptrue [[PG:p[0-9]+]].d
; CHECK-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG]]/m, z0.d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %res = fptrunc <2 x double> %op1 to <2 x half>
  ret <2 x half> %res
}

define <4 x half> @fcvt_v4f64_v4f16(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvt_v4f64_v4f16:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptrunc <4 x double> %op1 to <4 x half>
  ret <4 x half> %res
}

define <8 x half> @fcvt_v8f64_v8f16(<8 x double>* %a) #0 {
; CHECK-LABEL: fcvt_v8f64_v8f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[UZP]].h, [[UZP]].h
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: fcvt [[CVT_LO:z[0-9]+]].h, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: fcvt [[CVT_HI:z[0-9]+]].h, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: uzp1 [[UZP_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[UZP_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: uzp1 z0.h, [[UZP_LO]].h, [[UZP_LO]].h
; VBITS_EQ_256-DAG: uzp1 z[[RES_HI:[0-9]+]].h, [[UZP_HI]].h, [[UZP_HI]].h
; VBITS_EQ_256-DAG: mov v0.d[1], v[[RES_HI]].d[0]
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptrunc <8 x double> %op1 to <8 x half>
  ret <8 x half> %res
}

define void @fcvt_v16f64_v16f16(<16 x double>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v16f64_v16f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptrunc <16 x double> %op1 to <16 x half>
  store <16 x half> %res, <16 x half>* %b
  ret void
}

define void @fcvt_v32f64_v32f16(<32 x double>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fcvt_v32f64_v32f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvt [[CVT:z[0-9]+]].h, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[UZP:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[UZP]].h, [[UZP]].h
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].h, vl32
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptrunc <32 x double> %op1 to <32 x half>
  store <32 x half> %res, <32 x half>* %b
  ret void
}

;
; FCVT D -> S
;

; Don't use SVE for 64-bit vectors.
define <1 x float> @fcvt_v1f64_v1f32(<1 x double> %op1) #0 {
; CHECK-LABEL: fcvt_v1f64_v1f32:
; CHECK: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptrunc <1 x double> %op1 to <1 x float>
  ret <1 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x float> @fcvt_v2f64_v2f32(<2 x double> %op1) #0 {
; CHECK-LABEL: fcvt_v2f64_v2f32:
; CHECK: fcvtn v0.2s, v0.2d
; CHECK-NEXT: ret
  %res = fptrunc <2 x double> %op1 to <2 x float>
  ret <2 x float> %res
}

define <4 x float> @fcvt_v4f64_v4f32(<4 x double>* %a) #0 {
; CHECK-LABEL: fcvt_v4f64_v4f32:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: fcvt [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; CHECK-NEXT: uzp1 z0.s, [[CVT]].s, [[CVT]].s
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %res = fptrunc <4 x double> %op1 to <4 x float>
  ret <4 x float> %res
}

define void @fcvt_v8f64_v8f32(<8 x double>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v8f64_v8f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: fcvt [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_512-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_512-NEXT: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[LO:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[HI:z[0-9]+]].d }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].d
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].s, vl4
; VBITS_EQ_256-DAG: fcvt [[CVT_LO:z[0-9]+]].s, [[PG2]]/m, [[LO]].d
; VBITS_EQ_256-DAG: fcvt [[CVT_HI:z[0-9]+]].s, [[PG2]]/m, [[HI]].d
; VBITS_EQ_256-DAG: uzp1 [[RES_LO:z[0-9]+]].s, [[CVT_LO]].s, [[CVT_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[RES_HI:z[0-9]+]].s, [[CVT_HI]].s, [[CVT_HI]].s
; VBITS_EQ_256-DAG: splice [[RES:z[0-9]+]].s, [[PG3]], [[RES_LO]].s, [[RES_HI]].s
; VBITS_EQ_256-DAG: ptrue [[PG4:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[RES]].s }, [[PG4]], [x1]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %res = fptrunc <8 x double> %op1 to <8 x float>
  store <8 x float> %res, <8 x float>* %b
  ret void
}

define void @fcvt_v16f64_v16f32(<16 x double>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v16f64_v16f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: fcvt [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_1024-NEXT: ptrue [[PG3:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %res = fptrunc <16 x double> %op1 to <16 x float>
  store <16 x float> %res, <16 x float>* %b
  ret void
}

define void @fcvt_v32f64_v32f32(<32 x double>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fcvt_v32f64_v32f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: fcvt [[CVT:z[0-9]+]].s, [[PG2]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: uzp1 [[RES:z[0-9]+]].s, [[CVT]].s, [[CVT]].s
; VBITS_GE_2048-NEXT: ptrue [[PG3:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG3]], [x1]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %res = fptrunc <32 x double> %op1 to <32 x float>
  store <32 x float> %res, <32 x float>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
