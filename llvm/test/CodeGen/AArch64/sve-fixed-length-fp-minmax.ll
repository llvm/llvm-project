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
; FMAXNM
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fmaxnm_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fmaxnm_v4f16:
; CHECK: fmaxnm v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %op1, <4 x half> %op2)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fmaxnm_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fmaxnm_v8f16:
; CHECK: fmaxnm v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.maxnum.v8f16(<8 x half> %op1, <8 x half> %op2)
  ret <8 x half> %res
}

define void @fmaxnm_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fmaxnm_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: fmaxnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = call <16 x half> @llvm.maxnum.v16f16(<16 x half> %op1, <16 x half> %op2)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fmaxnm_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fmaxnm_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fmaxnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: fmaxnm [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: fmaxnm [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = call <32 x half> @llvm.maxnum.v32f16(<32 x half> %op1, <32 x half> %op2)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fmaxnm_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fmaxnm_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fmaxnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = call <64 x half> @llvm.maxnum.v64f16(<64 x half> %op1, <64 x half> %op2)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fmaxnm_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fmaxnm_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fmaxnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = call <128 x half> @llvm.maxnum.v128f16(<128 x half> %op1, <128 x half> %op2)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fmaxnm_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fmaxnm_v2f32:
; CHECK: fmaxnm v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %op1, <2 x float> %op2)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fmaxnm_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fmaxnm_v4f32:
; CHECK: fmaxnm v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %op1, <4 x float> %op2)
  ret <4 x float> %res
}

define void @fmaxnm_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fmaxnm_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: fmaxnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %op1, <8 x float> %op2)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fmaxnm_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fmaxnm_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fmaxnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: fmaxnm [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: fmaxnm [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = call <16 x float> @llvm.maxnum.v16f32(<16 x float> %op1, <16 x float> %op2)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fmaxnm_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fmaxnm_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fmaxnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = call <32 x float> @llvm.maxnum.v32f32(<32 x float> %op1, <32 x float> %op2)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fmaxnm_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fmaxnm_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fmaxnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = call <64 x float> @llvm.maxnum.v64f32(<64 x float> %op1, <64 x float> %op2)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fmaxnm_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fmaxnm_v1f64:
; CHECK: fmaxnm d0, d0, d1
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.maxnum.v1f64(<1 x double> %op1, <1 x double> %op2)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fmaxnm_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fmaxnm_v2f64:
; CHECK: fmaxnm v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %op1, <2 x double> %op2)
  ret <2 x double> %res
}

define void @fmaxnm_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fmaxnm_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: fmaxnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %op1, <4 x double> %op2)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fmaxnm_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fmaxnm_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fmaxnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: fmaxnm [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: fmaxnm [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = call <8 x double> @llvm.maxnum.v8f64(<8 x double> %op1, <8 x double> %op2)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fmaxnm_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fmaxnm_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fmaxnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = call <16 x double> @llvm.maxnum.v16f64(<16 x double> %op1, <16 x double> %op2)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fmaxnm_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fmaxnm_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fmaxnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = call <32 x double> @llvm.maxnum.v32f64(<32 x double> %op1, <32 x double> %op2)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FMINNM
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fminnm_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fminnm_v4f16:
; CHECK: fminnm v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = call <4 x half> @llvm.minnum.v4f16(<4 x half> %op1, <4 x half> %op2)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fminnm_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fminnm_v8f16:
; CHECK: fminnm v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %res = call <8 x half> @llvm.minnum.v8f16(<8 x half> %op1, <8 x half> %op2)
  ret <8 x half> %res
}

define void @fminnm_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fminnm_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: fminnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = call <16 x half> @llvm.minnum.v16f16(<16 x half> %op1, <16 x half> %op2)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fminnm_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fminnm_v32f16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fminnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
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
; VBITS_EQ_256-DAG: fminnm [[RES_LO:z[0-9]+]].h, [[PG]]/m, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: fminnm [[RES_HI:z[0-9]+]].h, [[PG]]/m, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: st1h { [[RES_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES_HI]].h }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = call <32 x half> @llvm.minnum.v32f16(<32 x half> %op1, <32 x half> %op2)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fminnm_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fminnm_v64f16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fminnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = call <64 x half> @llvm.minnum.v64f16(<64 x half> %op1, <64 x half> %op2)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fminnm_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fminnm_v128f16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fminnm [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = call <128 x half> @llvm.minnum.v128f16(<128 x half> %op1, <128 x half> %op2)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fminnm_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fminnm_v2f32:
; CHECK: fminnm v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = call <2 x float> @llvm.minnum.v2f32(<2 x float> %op1, <2 x float> %op2)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fminnm_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fminnm_v4f32:
; CHECK: fminnm v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %res = call <4 x float> @llvm.minnum.v4f32(<4 x float> %op1, <4 x float> %op2)
  ret <4 x float> %res
}

define void @fminnm_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fminnm_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: fminnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = call <8 x float> @llvm.minnum.v8f32(<8 x float> %op1, <8 x float> %op2)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fminnm_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fminnm_v16f32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fminnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
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
; VBITS_EQ_256-DAG: fminnm [[RES_LO:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: fminnm [[RES_HI:z[0-9]+]].s, [[PG]]/m, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: st1w { [[RES_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES_HI]].s }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = call <16 x float> @llvm.minnum.v16f32(<16 x float> %op1, <16 x float> %op2)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fminnm_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fminnm_v32f32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fminnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = call <32 x float> @llvm.minnum.v32f32(<32 x float> %op1, <32 x float> %op2)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fminnm_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fminnm_v64f32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fminnm [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = call <64 x float> @llvm.minnum.v64f32(<64 x float> %op1, <64 x float> %op2)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fminnm_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fminnm_v1f64:
; CHECK: fminnm d0, d0, d1
; CHECK-NEXT: ret
  %res = call <1 x double> @llvm.minnum.v1f64(<1 x double> %op1, <1 x double> %op2)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fminnm_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fminnm_v2f64:
; CHECK: fminnm v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %res = call <2 x double> @llvm.minnum.v2f64(<2 x double> %op1, <2 x double> %op2)
  ret <2 x double> %res
}

define void @fminnm_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fminnm_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: fminnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = call <4 x double> @llvm.minnum.v4f64(<4 x double> %op1, <4 x double> %op2)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fminnm_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fminnm_v8f64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: fminnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
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
; VBITS_EQ_256-DAG: fminnm [[RES_LO:z[0-9]+]].d, [[PG]]/m, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: fminnm [[RES_HI:z[0-9]+]].d, [[PG]]/m, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: st1d { [[RES_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES_HI]].d }, [[PG]], [x[[A_HI]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = call <8 x double> @llvm.minnum.v8f64(<8 x double> %op1, <8 x double> %op2)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fminnm_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fminnm_v16f64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: fminnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = call <16 x double> @llvm.minnum.v16f64(<16 x double> %op1, <16 x double> %op2)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fminnm_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fminnm_v32f64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: fminnm [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = call <32 x double> @llvm.minnum.v32f64(<32 x double> %op1, <32 x double> %op2)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

declare <4 x half> @llvm.minnum.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.minnum.v8f16(<8 x half>, <8 x half>)
declare <16 x half> @llvm.minnum.v16f16(<16 x half>, <16 x half>)
declare <32 x half> @llvm.minnum.v32f16(<32 x half>, <32 x half>)
declare <64 x half> @llvm.minnum.v64f16(<64 x half>, <64 x half>)
declare <128 x half> @llvm.minnum.v128f16(<128 x half>, <128 x half>)
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.minnum.v16f32(<16 x float>, <16 x float>)
declare <32 x float> @llvm.minnum.v32f32(<32 x float>, <32 x float>)
declare <64 x float> @llvm.minnum.v64f32(<64 x float>, <64 x float>)
declare <1 x double> @llvm.minnum.v1f64(<1 x double>, <1 x double>)
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>)
declare <16 x double> @llvm.minnum.v16f64(<16 x double>, <16 x double>)
declare <32 x double> @llvm.minnum.v32f64(<32 x double>, <32 x double>)

declare <4 x half> @llvm.maxnum.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.maxnum.v8f16(<8 x half>, <8 x half>)
declare <16 x half> @llvm.maxnum.v16f16(<16 x half>, <16 x half>)
declare <32 x half> @llvm.maxnum.v32f16(<32 x half>, <32 x half>)
declare <64 x half> @llvm.maxnum.v64f16(<64 x half>, <64 x half>)
declare <128 x half> @llvm.maxnum.v128f16(<128 x half>, <128 x half>)
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.maxnum.v16f32(<16 x float>, <16 x float>)
declare <32 x float> @llvm.maxnum.v32f32(<32 x float>, <32 x float>)
declare <64 x float> @llvm.maxnum.v64f32(<64 x float>, <64 x float>)
declare <1 x double> @llvm.maxnum.v1f64(<1 x double>, <1 x double>)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.maxnum.v8f64(<8 x double>, <8 x double>)
declare <16 x double> @llvm.maxnum.v16f64(<16 x double>, <16 x double>)
declare <32 x double> @llvm.maxnum.v32f64(<32 x double>, <32 x double>)
