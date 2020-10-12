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
; CEIL -> FRINTP
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frintp_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frintp_v4f16:
; CHECK: frintp v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.ceil.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frintp_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frintp_v8f16:
; CHECK: frintp v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.ceil.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frintp_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frintp_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frintp [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.ceil.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frintp_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frintp_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintp [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintp [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frintp [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.ceil.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frintp_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frintp_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintp [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.ceil.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frintp_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frintp_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintp [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.ceil.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frintp_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frintp_v2f32:
; CHECK: frintp v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.ceil.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frintp_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frintp_v4f32:
; CHECK: frintp v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.ceil.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frintp_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frintp_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frintp [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.ceil.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frintp_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frintp_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintp [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintp [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frintp [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.ceil.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frintp_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frintp_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintp [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.ceil.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frintp_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frintp_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintp [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.ceil.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frintp_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frintp_v1f64:
; CHECK: frintp d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.ceil.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frintp_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frintp_v2f64:
; CHECK: frintp v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.ceil.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frintp_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frintp_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frintp [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.ceil.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frintp_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frintp_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintp [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintp [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frintp [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.ceil.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frintp_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frintp_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintp [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.ceil.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frintp_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frintp_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintp [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.ceil.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FLOOR -> FRINTM
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frintm_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frintm_v4f16:
; CHECK: frintm v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.floor.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frintm_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frintm_v8f16:
; CHECK: frintm v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.floor.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frintm_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frintm_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frintm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.floor.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frintm_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frintm_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintm [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frintm [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.floor.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frintm_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frintm_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.floor.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frintm_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frintm_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.floor.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frintm_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frintm_v2f32:
; CHECK: frintm v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.floor.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frintm_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frintm_v4f32:
; CHECK: frintm v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.floor.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frintm_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frintm_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frintm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.floor.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frintm_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frintm_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintm [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frintm [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.floor.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frintm_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frintm_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.floor.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frintm_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frintm_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.floor.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frintm_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frintm_v1f64:
; CHECK: frintm d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.floor.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frintm_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frintm_v2f64:
; CHECK: frintm v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.floor.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frintm_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frintm_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frintm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.floor.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frintm_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frintm_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintm [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frintm [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.floor.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frintm_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frintm_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.floor.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frintm_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frintm_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.floor.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FNEARBYINT -> FRINTI
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frinti_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frinti_v4f16:
; CHECK: frinti v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.nearbyint.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frinti_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frinti_v8f16:
; CHECK: frinti v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.nearbyint.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frinti_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frinti_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frinti [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.nearbyint.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frinti_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frinti_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinti [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinti [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frinti [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.nearbyint.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frinti_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frinti_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinti [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.nearbyint.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frinti_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frinti_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinti [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.nearbyint.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frinti_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frinti_v2f32:
; CHECK: frinti v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frinti_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frinti_v4f32:
; CHECK: frinti v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frinti_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frinti_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frinti [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.nearbyint.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frinti_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frinti_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinti [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinti [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frinti [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.nearbyint.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frinti_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frinti_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinti [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.nearbyint.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frinti_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frinti_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinti [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.nearbyint.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frinti_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frinti_v1f64:
; CHECK: frinti d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.nearbyint.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frinti_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frinti_v2f64:
; CHECK: frinti v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frinti_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frinti_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frinti [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frinti_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frinti_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinti [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinti [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frinti [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.nearbyint.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frinti_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frinti_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinti [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.nearbyint.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frinti_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frinti_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinti [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.nearbyint.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; RINT -> FRINTX
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frintx_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frintx_v4f16:
; CHECK: frintx v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.rint.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frintx_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frintx_v8f16:
; CHECK: frintx v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.rint.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frintx_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frintx_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frintx [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.rint.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frintx_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frintx_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintx [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintx [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frintx [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.rint.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frintx_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frintx_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintx [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.rint.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frintx_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frintx_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintx [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.rint.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frintx_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frintx_v2f32:
; CHECK: frintx v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.rint.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frintx_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frintx_v4f32:
; CHECK: frintx v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.rint.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frintx_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frintx_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frintx [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.rint.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frintx_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frintx_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintx [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintx [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frintx [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.rint.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frintx_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frintx_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintx [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.rint.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frintx_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frintx_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintx [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.rint.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frintx_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frintx_v1f64:
; CHECK: frintx d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.rint.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frintx_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frintx_v2f64:
; CHECK: frintx v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.rint.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frintx_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frintx_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frintx [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.rint.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frintx_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frintx_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintx [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintx [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frintx [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.rint.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frintx_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frintx_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintx [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.rint.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frintx_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frintx_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintx [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.rint.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; ROUND -> FRINTA
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frinta_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frinta_v4f16:
; CHECK: frinta v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.round.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frinta_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frinta_v8f16:
; CHECK: frinta v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.round.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frinta_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frinta_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frinta [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.round.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frinta_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frinta_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinta [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinta [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frinta [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.round.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frinta_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frinta_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinta [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.round.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frinta_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frinta_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinta [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.round.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frinta_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frinta_v2f32:
; CHECK: frinta v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.round.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frinta_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frinta_v4f32:
; CHECK: frinta v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.round.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frinta_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frinta_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frinta [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.round.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frinta_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frinta_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinta [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinta [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frinta [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.round.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frinta_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frinta_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinta [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.round.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frinta_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frinta_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinta [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.round.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frinta_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frinta_v1f64:
; CHECK: frinta d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.round.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frinta_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frinta_v2f64:
; CHECK: frinta v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.round.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frinta_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frinta_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frinta [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.round.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frinta_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frinta_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frinta [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frinta [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frinta [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.round.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frinta_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frinta_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frinta [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.round.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frinta_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frinta_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frinta [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.round.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; TRUNC -> FRINTZ
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @frintz_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: frintz_v4f16:
; CHECK: frintz v0.4h, v0.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.trunc.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @frintz_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: frintz_v8f16:
; CHECK: frintz v0.8h, v0.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.trunc.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @frintz_v16f16(<16 x half>* %a) #0 {
; CHECK-LABEL: frintz_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: frintz [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.trunc.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @frintz_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: frintz_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintz [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1h { [[OP_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintz [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP_LO]].h
; VBITS_EQ_256-DAG: frintz [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.trunc.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @frintz_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: frintz_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintz [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.trunc.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @frintz_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: frintz_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintz [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.trunc.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @frintz_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: frintz_v2f32:
; CHECK: frintz v0.2s, v0.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.trunc.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @frintz_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: frintz_v4f32:
; CHECK: frintz v0.4s, v0.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.trunc.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @frintz_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: frintz_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: frintz [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.trunc.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @frintz_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: frintz_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintz [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1w { [[OP_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintz [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP_LO]].s
; VBITS_EQ_256-DAG: frintz [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.trunc.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @frintz_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: frintz_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintz [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.trunc.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @frintz_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: frintz_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintz [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.trunc.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @frintz_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: frintz_v1f64:
; CHECK: frintz d0, d0
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.trunc.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @frintz_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: frintz_v2f64:
; CHECK: frintz v0.2d, v0.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.trunc.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @frintz_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: frintz_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: frintz [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.trunc.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @frintz_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: frintz_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: frintz [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: ld1d { [[OP_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: frintz [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP_LO]].d
; VBITS_EQ_256-DAG: frintz [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.trunc.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @frintz_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: frintz_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: frintz [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.trunc.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @frintz_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: frintz_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: frintz [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.trunc.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

declare <4 x half> @llvm.ceil.v4f16(<4 x half>)
declare <8 x half> @llvm.ceil.v8f16(<8 x half>)
declare <16 x half> @llvm.ceil.v16f16(<16 x half>)
declare <32 x half> @llvm.ceil.v32f16(<32 x half>)
declare <64 x half> @llvm.ceil.v64f16(<64 x half>)
declare <128 x half> @llvm.ceil.v128f16(<128 x half>)
declare <2 x float> @llvm.ceil.v2f32(<2 x float>)
declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
declare <8 x float> @llvm.ceil.v8f32(<8 x float>)
declare <16 x float> @llvm.ceil.v16f32(<16 x float>)
declare <32 x float> @llvm.ceil.v32f32(<32 x float>)
declare <64 x float> @llvm.ceil.v64f32(<64 x float>)
declare <1 x double> @llvm.ceil.v1f64(<1 x double>)
declare <2 x double> @llvm.ceil.v2f64(<2 x double>)
declare <4 x double> @llvm.ceil.v4f64(<4 x double>)
declare <8 x double> @llvm.ceil.v8f64(<8 x double>)
declare <16 x double> @llvm.ceil.v16f64(<16 x double>)
declare <32 x double> @llvm.ceil.v32f64(<32 x double>)

declare <4 x half> @llvm.floor.v4f16(<4 x half>)
declare <8 x half> @llvm.floor.v8f16(<8 x half>)
declare <16 x half> @llvm.floor.v16f16(<16 x half>)
declare <32 x half> @llvm.floor.v32f16(<32 x half>)
declare <64 x half> @llvm.floor.v64f16(<64 x half>)
declare <128 x half> @llvm.floor.v128f16(<128 x half>)
declare <2 x float> @llvm.floor.v2f32(<2 x float>)
declare <4 x float> @llvm.floor.v4f32(<4 x float>)
declare <8 x float> @llvm.floor.v8f32(<8 x float>)
declare <16 x float> @llvm.floor.v16f32(<16 x float>)
declare <32 x float> @llvm.floor.v32f32(<32 x float>)
declare <64 x float> @llvm.floor.v64f32(<64 x float>)
declare <1 x double> @llvm.floor.v1f64(<1 x double>)
declare <2 x double> @llvm.floor.v2f64(<2 x double>)
declare <4 x double> @llvm.floor.v4f64(<4 x double>)
declare <8 x double> @llvm.floor.v8f64(<8 x double>)
declare <16 x double> @llvm.floor.v16f64(<16 x double>)
declare <32 x double> @llvm.floor.v32f64(<32 x double>)

declare <4 x half> @llvm.nearbyint.v4f16(<4 x half>)
declare <8 x half> @llvm.nearbyint.v8f16(<8 x half>)
declare <16 x half> @llvm.nearbyint.v16f16(<16 x half>)
declare <32 x half> @llvm.nearbyint.v32f16(<32 x half>)
declare <64 x half> @llvm.nearbyint.v64f16(<64 x half>)
declare <128 x half> @llvm.nearbyint.v128f16(<128 x half>)
declare <2 x float> @llvm.nearbyint.v2f32(<2 x float>)
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>)
declare <8 x float> @llvm.nearbyint.v8f32(<8 x float>)
declare <16 x float> @llvm.nearbyint.v16f32(<16 x float>)
declare <32 x float> @llvm.nearbyint.v32f32(<32 x float>)
declare <64 x float> @llvm.nearbyint.v64f32(<64 x float>)
declare <1 x double> @llvm.nearbyint.v1f64(<1 x double>)
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>)
declare <4 x double> @llvm.nearbyint.v4f64(<4 x double>)
declare <8 x double> @llvm.nearbyint.v8f64(<8 x double>)
declare <16 x double> @llvm.nearbyint.v16f64(<16 x double>)
declare <32 x double> @llvm.nearbyint.v32f64(<32 x double>)

declare <4 x half> @llvm.rint.v4f16(<4 x half>)
declare <8 x half> @llvm.rint.v8f16(<8 x half>)
declare <16 x half> @llvm.rint.v16f16(<16 x half>)
declare <32 x half> @llvm.rint.v32f16(<32 x half>)
declare <64 x half> @llvm.rint.v64f16(<64 x half>)
declare <128 x half> @llvm.rint.v128f16(<128 x half>)
declare <2 x float> @llvm.rint.v2f32(<2 x float>)
declare <4 x float> @llvm.rint.v4f32(<4 x float>)
declare <8 x float> @llvm.rint.v8f32(<8 x float>)
declare <16 x float> @llvm.rint.v16f32(<16 x float>)
declare <32 x float> @llvm.rint.v32f32(<32 x float>)
declare <64 x float> @llvm.rint.v64f32(<64 x float>)
declare <1 x double> @llvm.rint.v1f64(<1 x double>)
declare <2 x double> @llvm.rint.v2f64(<2 x double>)
declare <4 x double> @llvm.rint.v4f64(<4 x double>)
declare <8 x double> @llvm.rint.v8f64(<8 x double>)
declare <16 x double> @llvm.rint.v16f64(<16 x double>)
declare <32 x double> @llvm.rint.v32f64(<32 x double>)

declare <4 x half> @llvm.round.v4f16(<4 x half>)
declare <8 x half> @llvm.round.v8f16(<8 x half>)
declare <16 x half> @llvm.round.v16f16(<16 x half>)
declare <32 x half> @llvm.round.v32f16(<32 x half>)
declare <64 x half> @llvm.round.v64f16(<64 x half>)
declare <128 x half> @llvm.round.v128f16(<128 x half>)
declare <2 x float> @llvm.round.v2f32(<2 x float>)
declare <4 x float> @llvm.round.v4f32(<4 x float>)
declare <8 x float> @llvm.round.v8f32(<8 x float>)
declare <16 x float> @llvm.round.v16f32(<16 x float>)
declare <32 x float> @llvm.round.v32f32(<32 x float>)
declare <64 x float> @llvm.round.v64f32(<64 x float>)
declare <1 x double> @llvm.round.v1f64(<1 x double>)
declare <2 x double> @llvm.round.v2f64(<2 x double>)
declare <4 x double> @llvm.round.v4f64(<4 x double>)
declare <8 x double> @llvm.round.v8f64(<8 x double>)
declare <16 x double> @llvm.round.v16f64(<16 x double>)
declare <32 x double> @llvm.round.v32f64(<32 x double>)

declare <4 x half> @llvm.trunc.v4f16(<4 x half>)
declare <8 x half> @llvm.trunc.v8f16(<8 x half>)
declare <16 x half> @llvm.trunc.v16f16(<16 x half>)
declare <32 x half> @llvm.trunc.v32f16(<32 x half>)
declare <64 x half> @llvm.trunc.v64f16(<64 x half>)
declare <128 x half> @llvm.trunc.v128f16(<128 x half>)
declare <2 x float> @llvm.trunc.v2f32(<2 x float>)
declare <4 x float> @llvm.trunc.v4f32(<4 x float>)
declare <8 x float> @llvm.trunc.v8f32(<8 x float>)
declare <16 x float> @llvm.trunc.v16f32(<16 x float>)
declare <32 x float> @llvm.trunc.v32f32(<32 x float>)
declare <64 x float> @llvm.trunc.v64f32(<64 x float>)
declare <1 x double> @llvm.trunc.v1f64(<1 x double>)
declare <2 x double> @llvm.trunc.v2f64(<2 x double>)
declare <4 x double> @llvm.trunc.v4f64(<4 x double>)
declare <8 x double> @llvm.trunc.v8f64(<8 x double>)
declare <16 x double> @llvm.trunc.v16f64(<16 x double>)
declare <32 x double> @llvm.trunc.v32f64(<32 x double>)
