; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512,VBITS_LE_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512,VBITS_LE_256
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_LE_1024,VBITS_LE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_LE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK

; VBYTES represents the useful byte size of a vector register from the code
; generator's point of view. It is clamped to power-of-2 values because
; only power-of-2 vector lengths are considered legal, regardless of the
; user specified vector length.

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; FADD
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fadd_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fadd_v4f16:
; CHECK: fadd v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = fadd <4 x half> %op1, %op2
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fadd_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fadd_v8f16:
; CHECK: fadd v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = fadd <8 x half> %op1, %op2
  ret <8 x half> %res
}

define void @fadd_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fadd_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = fadd <16 x half> %op1, %op2
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fadd_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fadd_v32f16:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: fadd [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-DAG: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_LE_256-DAG: add x[[A1:[0-9]+]], x0, #[[#VBYTES]]
; VBITS_LE_256-DAG: add x[[B1:[0-9]+]], x1, #[[#VBYTES]]
; VBITS_LE_256-DAG: ld1h { [[OP1_1:z[0-9]+]].h }, [[PG]]/z, [x[[A1]]]
; VBITS_LE_256-DAG: ld1h { [[OP2_1:z[0-9]+]].h }, [[PG]]/z, [x[[B1]]]
; VBITS_LE_256-DAG: fadd [[RES_1:z[0-9]+]].h, [[PG]]/m, [[OP1_1]].h, [[OP2_1]].h
; VBITS_LE_256-DAG: st1h { [[RES_1]].h }, [[PG]], [x[[A1]]]
; CHECK: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = fadd <32 x half> %op1, %op2
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fadd_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fadd_v64f16:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: fadd [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK-DAG: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_LE_512-DAG: add x[[A1:[0-9]+]], x0, #[[#VBYTES]]
; VBITS_LE_512-DAG: add x[[B1:[0-9]+]], x1, #[[#VBYTES]]
; VBITS_LE_512-DAG: ld1h { [[OP1_1:z[0-9]+]].h }, [[PG]]/z, [x[[A1]]]
; VBITS_LE_512-DAG: ld1h { [[OP2_1:z[0-9]+]].h }, [[PG]]/z, [x[[B1]]]
; VBITS_LE_512-DAG: fadd [[RES_1:z[0-9]+]].h, [[PG]]/m, [[OP1_1]].h, [[OP2_1]].h
; VBITS_LE_512-DAG: st1h { [[RES_1]].h }, [[PG]], [x[[A1]]]
; VBITS_LE_256-DAG: add x[[A2:[0-9]+]], x0, #[[#mul(VBYTES,2)]]
; VBITS_LE_256-DAG: add x[[B2:[0-9]+]], x1, #[[#mul(VBYTES,2)]]
; VBITS_LE_256-DAG: ld1h { [[OP1_2:z[0-9]+]].h }, [[PG]]/z, [x[[A2]]]
; VBITS_LE_256-DAG: ld1h { [[OP2_2:z[0-9]+]].h }, [[PG]]/z, [x[[B2]]]
; VBITS_LE_256-DAG: fadd [[RES_2:z[0-9]+]].h, [[PG]]/m, [[OP1_2]].h, [[OP2_2]].h
; VBITS_LE_256-DAG: st1h { [[RES_2]].h }, [[PG]], [x[[A2]]]
; VBITS_LE_256-DAG: add x[[A3:[0-9]+]], x0, #[[#mul(VBYTES,3)]]
; VBITS_LE_256-DAG: add x[[B3:[0-9]+]], x1, #[[#mul(VBYTES,3)]]
; VBITS_LE_256-DAG: ld1h { [[OP1_3:z[0-9]+]].h }, [[PG]]/z, [x[[A3]]]
; VBITS_LE_256-DAG: ld1h { [[OP2_3:z[0-9]+]].h }, [[PG]]/z, [x[[B3]]]
; VBITS_LE_256-DAG: fadd [[RES_3:z[0-9]+]].h, [[PG]]/m, [[OP1_3]].h, [[OP2_3]].h
; VBITS_LE_256-DAG: st1h { [[RES_3]].h }, [[PG]], [x[[A3]]]
; CHECK: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = fadd <64 x half> %op1, %op2
  store <64 x half> %res, <64 x half>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fadd_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = fadd <128 x half> %op1, %op2
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fadd_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fadd_v2f32:
; CHECK: fadd v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = fadd <2 x float> %op1, %op2
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fadd_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fadd_v4f32:
; CHECK: fadd v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = fadd <4 x float> %op1, %op2
  ret <4 x float> %res
}

define void @fadd_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fadd_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = fadd <8 x float> %op1, %op2
  store <8 x float> %res, <8 x float>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fadd_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = fadd <16 x float> %op1, %op2
  store <16 x float> %res, <16 x float>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fadd_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = fadd <32 x float> %op1, %op2
  store <32 x float> %res, <32 x float>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fadd_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = fadd <64 x float> %op1, %op2
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fadd_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fadd_v1f64:
; CHECK: fadd d0, d0, d1
; CHECK: ret
  %res = fadd <1 x double> %op1, %op2
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fadd_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fadd_v2f64:
; CHECK: fadd v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = fadd <2 x double> %op1, %op2
  ret <2 x double> %res
}

define void @fadd_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fadd_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = fadd <4 x double> %op1, %op2
  store <4 x double> %res, <4 x double>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fadd_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = fadd <8 x double> %op1, %op2
  store <8 x double> %res, <8 x double>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fadd_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = fadd <16 x double> %op1, %op2
  store <16 x double> %res, <16 x double>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the fadd_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fadd_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fadd [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = fadd <32 x double> %op1, %op2
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; NOTE: Tests beyond this point only have CHECK lines to validate the first
; VBYTES because the fadd tests already validate the legalisation code paths.
;

;
; FDIV
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fdiv_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fdiv_v4f16:
; CHECK: fdiv v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = fdiv <4 x half> %op1, %op2
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fdiv_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fdiv_v8f16:
; CHECK: fdiv v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = fdiv <8 x half> %op1, %op2
  ret <8 x half> %res
}

define void @fdiv_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fdiv_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = fdiv <16 x half> %op1, %op2
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fdiv_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fdiv_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = fdiv <32 x half> %op1, %op2
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fdiv_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fdiv_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = fdiv <64 x half> %op1, %op2
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fdiv_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fdiv_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = fdiv <128 x half> %op1, %op2
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fdiv_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fdiv_v2f32:
; CHECK: fdiv v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = fdiv <2 x float> %op1, %op2
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fdiv_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fdiv_v4f32:
; CHECK: fdiv v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = fdiv <4 x float> %op1, %op2
  ret <4 x float> %res
}

define void @fdiv_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fdiv_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = fdiv <8 x float> %op1, %op2
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fdiv_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fdiv_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = fdiv <16 x float> %op1, %op2
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fdiv_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fdiv_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = fdiv <32 x float> %op1, %op2
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fdiv_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fdiv_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = fdiv <64 x float> %op1, %op2
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fdiv_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fdiv_v1f64:
; CHECK: fdiv d0, d0, d1
; CHECK: ret
  %res = fdiv <1 x double> %op1, %op2
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fdiv_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fdiv_v2f64:
; CHECK: fdiv v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = fdiv <2 x double> %op1, %op2
  ret <2 x double> %res
}

define void @fdiv_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fdiv_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = fdiv <4 x double> %op1, %op2
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fdiv_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fdiv_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = fdiv <8 x double> %op1, %op2
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fdiv_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fdiv_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = fdiv <16 x double> %op1, %op2
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fdiv_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fdiv_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = fdiv <32 x double> %op1, %op2
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FMA
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fma_v4f16(<4 x half> %op1, <4 x half> %op2, <4 x half> %op3) #0 {
; CHECK-LABEL: fma_v4f16:
; CHECK: fmla v2.4h, v1.4h, v0.4h
; CHECK: ret
  %res = call <4 x half> @llvm.fma.v4f16(<4 x half> %op1, <4 x half> %op2, <4 x half> %op3)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fma_v8f16(<8 x half> %op1, <8 x half> %op2, <8 x half> %op3) #0 {
; CHECK-LABEL: fma_v8f16:
; CHECK: fmla v2.8h, v1.8h, v0.8h
; CHECK: ret
  %res = call <8 x half> @llvm.fma.v8f16(<8 x half> %op1, <8 x half> %op2, <8 x half> %op3)
  ret <8 x half> %res
}

define void @fma_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x half>* %c) #0 {
; CHECK-LABEL: fma_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: ld1h { [[OP3:z[0-9]+]].h }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[OP3]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %op3 = load <16 x half>, <16 x half>* %c
  %res = call <16 x half> @llvm.fma.v16f16(<16 x half> %op1, <16 x half> %op2, <16 x half> %op3)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fma_v32f16(<32 x half>* %a, <32 x half>* %b, <32 x half>* %c) #0 {
; CHECK-LABEL: fma_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: ld1h { [[OP3:z[0-9]+]].h }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[OP3]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %op3 = load <32 x half>, <32 x half>* %c
  %res = call <32 x half> @llvm.fma.v32f16(<32 x half> %op1, <32 x half> %op2, <32 x half> %op3)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fma_v64f16(<64 x half>* %a, <64 x half>* %b, <64 x half>* %c) #0 {
; CHECK-LABEL: fma_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: ld1h { [[OP3:z[0-9]+]].h }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[OP3]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %op3 = load <64 x half>, <64 x half>* %c
  %res = call <64 x half> @llvm.fma.v64f16(<64 x half> %op1, <64 x half> %op2, <64 x half> %op3)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fma_v128f16(<128 x half>* %a, <128 x half>* %b, <128 x half>* %c) #0 {
; CHECK-LABEL: fma_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-DAG: ld1h { [[OP3:z[0-9]+]].h }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[OP3]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %op3 = load <128 x half>, <128 x half>* %c
  %res = call <128 x half> @llvm.fma.v128f16(<128 x half> %op1, <128 x half> %op2, <128 x half> %op3)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fma_v2f32(<2 x float> %op1, <2 x float> %op2, <2 x float> %op3) #0 {
; CHECK-LABEL: fma_v2f32:
; CHECK: fmla v2.2s, v1.2s, v0.2s
; CHECK: ret
  %res = call <2 x float> @llvm.fma.v2f32(<2 x float> %op1, <2 x float> %op2, <2 x float> %op3)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fma_v4f32(<4 x float> %op1, <4 x float> %op2, <4 x float> %op3) #0 {
; CHECK-LABEL: fma_v4f32:
; CHECK: fmla v2.4s, v1.4s, v0.4s
; CHECK: ret
  %res = call <4 x float> @llvm.fma.v4f32(<4 x float> %op1, <4 x float> %op2, <4 x float> %op3)
  ret <4 x float> %res
}

define void @fma_v8f32(<8 x float>* %a, <8 x float>* %b, <8 x float>* %c) #0 {
; CHECK-LABEL: fma_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-DAG: ld1w { [[OP3:z[0-9]+]].s }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[OP3]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %op3 = load <8 x float>, <8 x float>* %c
  %res = call <8 x float> @llvm.fma.v8f32(<8 x float> %op1, <8 x float> %op2, <8 x float> %op3)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fma_v16f32(<16 x float>* %a, <16 x float>* %b, <16 x float>* %c) #0 {
; CHECK-LABEL: fma_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-DAG: ld1w { [[OP3:z[0-9]+]].s }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[OP3]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %op3 = load <16 x float>, <16 x float>* %c
  %res = call <16 x float> @llvm.fma.v16f32(<16 x float> %op1, <16 x float> %op2, <16 x float> %op3)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fma_v32f32(<32 x float>* %a, <32 x float>* %b, <32 x float>* %c) #0 {
; CHECK-LABEL: fma_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-DAG: ld1w { [[OP3:z[0-9]+]].s }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[OP3]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %op3 = load <32 x float>, <32 x float>* %c
  %res = call <32 x float> @llvm.fma.v32f32(<32 x float> %op1, <32 x float> %op2, <32 x float> %op3)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fma_v64f32(<64 x float>* %a, <64 x float>* %b, <64 x float>* %c) #0 {
; CHECK-LABEL: fma_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-DAG: ld1w { [[OP3:z[0-9]+]].s }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[OP3]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %op3 = load <64 x float>, <64 x float>* %c
  %res = call <64 x float> @llvm.fma.v64f32(<64 x float> %op1, <64 x float> %op2, <64 x float> %op3)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fma_v1f64(<1 x double> %op1, <1 x double> %op2, <1 x double> %op3) #0 {
; CHECK-LABEL: fma_v1f64:
; CHECK: fmadd d0, d0, d1, d2
; CHECK: ret
  %res = call <1 x double> @llvm.fma.v1f64(<1 x double> %op1, <1 x double> %op2, <1 x double> %op3)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fma_v2f64(<2 x double> %op1, <2 x double> %op2, <2 x double> %op3) #0 {
; CHECK-LABEL: fma_v2f64:
; CHECK: fmla v2.2d, v1.2d, v0.2d
; CHECK: ret
  %res = call <2 x double> @llvm.fma.v2f64(<2 x double> %op1, <2 x double> %op2, <2 x double> %op3)
  ret <2 x double> %res
}

define void @fma_v4f64(<4 x double>* %a, <4 x double>* %b, <4 x double>* %c) #0 {
; CHECK-LABEL: fma_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-DAG: ld1d { [[OP3:z[0-9]+]].d }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[OP3]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %op3 = load <4 x double>, <4 x double>* %c
  %res = call <4 x double> @llvm.fma.v4f64(<4 x double> %op1, <4 x double> %op2, <4 x double> %op3)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fma_v8f64(<8 x double>* %a, <8 x double>* %b, <8 x double>* %c) #0 {
; CHECK-LABEL: fma_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-DAG: ld1d { [[OP3:z[0-9]+]].d }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[OP3]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %op3 = load <8 x double>, <8 x double>* %c
  %res = call <8 x double> @llvm.fma.v8f64(<8 x double> %op1, <8 x double> %op2, <8 x double> %op3)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fma_v16f64(<16 x double>* %a, <16 x double>* %b, <16 x double>* %c) #0 {
; CHECK-LABEL: fma_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-DAG: ld1d { [[OP3:z[0-9]+]].d }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[OP3]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %op3 = load <16 x double>, <16 x double>* %c
  %res = call <16 x double> @llvm.fma.v16f64(<16 x double> %op1, <16 x double> %op2, <16 x double> %op3)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fma_v32f64(<32 x double>* %a, <32 x double>* %b, <32 x double>* %c) #0 {
; CHECK-LABEL: fma_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-DAG: ld1d { [[OP3:z[0-9]+]].d }, [[PG]]/z, [x2]
; CHECK: fmla [[OP3]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[OP3]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %op3 = load <32 x double>, <32 x double>* %c
  %res = call <32 x double> @llvm.fma.v32f64(<32 x double> %op1, <32 x double> %op2, <32 x double> %op3)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FMUL
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fmul_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fmul_v4f16:
; CHECK: fmul v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = fmul <4 x half> %op1, %op2
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fmul_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fmul_v8f16:
; CHECK: fmul v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = fmul <8 x half> %op1, %op2
  ret <8 x half> %res
}

define void @fmul_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fmul_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = fmul <16 x half> %op1, %op2
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fmul_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fmul_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = fmul <32 x half> %op1, %op2
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fmul_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fmul_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = fmul <64 x half> %op1, %op2
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fmul_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fmul_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = fmul <128 x half> %op1, %op2
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fmul_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fmul_v2f32:
; CHECK: fmul v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = fmul <2 x float> %op1, %op2
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fmul_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fmul_v4f32:
; CHECK: fmul v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = fmul <4 x float> %op1, %op2
  ret <4 x float> %res
}

define void @fmul_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fmul_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = fmul <8 x float> %op1, %op2
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fmul_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fmul_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = fmul <16 x float> %op1, %op2
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fmul_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fmul_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = fmul <32 x float> %op1, %op2
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fmul_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fmul_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = fmul <64 x float> %op1, %op2
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fmul_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fmul_v1f64:
; CHECK: fmul d0, d0, d1
; CHECK: ret
  %res = fmul <1 x double> %op1, %op2
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fmul_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fmul_v2f64:
; CHECK: fmul v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = fmul <2 x double> %op1, %op2
  ret <2 x double> %res
}

define void @fmul_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fmul_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = fmul <4 x double> %op1, %op2
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fmul_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fmul_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = fmul <8 x double> %op1, %op2
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fmul_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fmul_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = fmul <16 x double> %op1, %op2
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fmul_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fmul_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fmul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = fmul <32 x double> %op1, %op2
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FNEG
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fneg_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: fneg_v4f16:
; CHECK: fneg v0.4h, v0.4h
; CHECK: ret
  %res = fneg <4 x half> %op
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fneg_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: fneg_v8f16:
; CHECK: fneg v0.8h, v0.8h
; CHECK: ret
  %res = fneg <8 x half> %op
  ret <8 x half> %res
}

define void @fneg_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fneg_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = fneg <16 x half> %op
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fneg_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: fneg_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = fneg <32 x half> %op
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fneg_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: fneg_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = fneg <64 x half> %op
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fneg_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: fneg_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = fneg <128 x half> %op
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fneg_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: fneg_v2f32:
; CHECK: fneg v0.2s, v0.2s
; CHECK: ret
  %res = fneg <2 x float> %op
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fneg_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: fneg_v4f32:
; CHECK: fneg v0.4s, v0.4s
; CHECK: ret
  %res = fneg <4 x float> %op
  ret <4 x float> %res
}

define void @fneg_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: fneg_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = fneg <8 x float> %op
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fneg_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: fneg_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = fneg <16 x float> %op
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fneg_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: fneg_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = fneg <32 x float> %op
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fneg_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: fneg_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = fneg <64 x float> %op
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fneg_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: fneg_v1f64:
; CHECK: fneg d0, d0
; CHECK: ret
  %res = fneg <1 x double> %op
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fneg_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: fneg_v2f64:
; CHECK: fneg v0.2d, v0.2d
; CHECK: ret
  %res = fneg <2 x double> %op
  ret <2 x double> %res
}

define void @fneg_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: fneg_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = fneg <4 x double> %op
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fneg_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: fneg_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = fneg <8 x double> %op
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fneg_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: fneg_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = fneg <16 x double> %op
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fneg_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: fneg_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fneg [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = fneg <32 x double> %op
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FSQRT
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fsqrt_v4f16(<4 x half> %op) #0 {
; CHECK-LABEL: fsqrt_v4f16:
; CHECK: fsqrt v0.4h, v0.4h
; CHECK: ret
  %res = call <4 x half> @llvm.sqrt.v4f16(<4 x half> %op)
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fsqrt_v8f16(<8 x half> %op) #0 {
; CHECK-LABEL: fsqrt_v8f16:
; CHECK: fsqrt v0.8h, v0.8h
; CHECK: ret
  %res = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %op)
  ret <8 x half> %res
}

define void @fsqrt_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fsqrt_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x half>, <16 x half>* %a
  %res = call <16 x half> @llvm.sqrt.v16f16(<16 x half> %op)
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fsqrt_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: fsqrt_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x half>, <32 x half>* %a
  %res = call <32 x half> @llvm.sqrt.v32f16(<32 x half> %op)
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fsqrt_v64f16(<64 x half>* %a) #0 {
; CHECK-LABEL: fsqrt_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <64 x half>, <64 x half>* %a
  %res = call <64 x half> @llvm.sqrt.v64f16(<64 x half> %op)
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fsqrt_v128f16(<128 x half>* %a) #0 {
; CHECK-LABEL: fsqrt_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK: ld1h { [[OP:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].h, [[PG]]/m, [[OP]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op = load <128 x half>, <128 x half>* %a
  %res = call <128 x half> @llvm.sqrt.v128f16(<128 x half> %op)
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fsqrt_v2f32(<2 x float> %op) #0 {
; CHECK-LABEL: fsqrt_v2f32:
; CHECK: fsqrt v0.2s, v0.2s
; CHECK: ret
  %res = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %op)
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fsqrt_v4f32(<4 x float> %op) #0 {
; CHECK-LABEL: fsqrt_v4f32:
; CHECK: fsqrt v0.4s, v0.4s
; CHECK: ret
  %res = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %op)
  ret <4 x float> %res
}

define void @fsqrt_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: fsqrt_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <8 x float>, <8 x float>* %a
  %res = call <8 x float> @llvm.sqrt.v8f32(<8 x float> %op)
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fsqrt_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: fsqrt_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x float>, <16 x float>* %a
  %res = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %op)
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fsqrt_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: fsqrt_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x float>, <32 x float>* %a
  %res = call <32 x float> @llvm.sqrt.v32f32(<32 x float> %op)
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fsqrt_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: fsqrt_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK: ld1w { [[OP:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].s, [[PG]]/m, [[OP]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op = load <64 x float>, <64 x float>* %a
  %res = call <64 x float> @llvm.sqrt.v64f32(<64 x float> %op)
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fsqrt_v1f64(<1 x double> %op) #0 {
; CHECK-LABEL: fsqrt_v1f64:
; CHECK: fsqrt d0, d0
; CHECK: ret
  %res = call <1 x double> @llvm.sqrt.v1f64(<1 x double> %op)
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fsqrt_v2f64(<2 x double> %op) #0 {
; CHECK-LABEL: fsqrt_v2f64:
; CHECK: fsqrt v0.2d, v0.2d
; CHECK: ret
  %res = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %op)
  ret <2 x double> %res
}

define void @fsqrt_v4f64(<4 x double>* %a) #0 {
; CHECK-LABEL: fsqrt_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <4 x double>, <4 x double>* %a
  %res = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %op)
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fsqrt_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: fsqrt_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <8 x double>, <8 x double>* %a
  %res = call <8 x double> @llvm.sqrt.v8f64(<8 x double> %op)
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fsqrt_v16f64(<16 x double>* %a) #0 {
; CHECK-LABEL: fsqrt_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <16 x double>, <16 x double>* %a
  %res = call <16 x double> @llvm.sqrt.v16f64(<16 x double> %op)
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fsqrt_v32f64(<32 x double>* %a) #0 {
; CHECK-LABEL: fsqrt_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK: ld1d { [[OP:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: fsqrt [[RES:z[0-9]+]].d, [[PG]]/m, [[OP]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op = load <32 x double>, <32 x double>* %a
  %res = call <32 x double> @llvm.sqrt.v32f64(<32 x double> %op)
  store <32 x double> %res, <32 x double>* %a
  ret void
}

;
; FSUB
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @fsub_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: fsub_v4f16:
; CHECK: fsub v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = fsub <4 x half> %op1, %op2
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fsub_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: fsub_v8f16:
; CHECK: fsub v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = fsub <8 x half> %op1, %op2
  ret <8 x half> %res
}

define void @fsub_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: fsub_v16f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = fsub <16 x half> %op1, %op2
  store <16 x half> %res, <16 x half>* %a
  ret void
}

define void @fsub_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; CHECK-LABEL: fsub_v32f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = fsub <32 x half> %op1, %op2
  store <32 x half> %res, <32 x half>* %a
  ret void
}

define void @fsub_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; CHECK-LABEL: fsub_v64f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = fsub <64 x half> %op1, %op2
  store <64 x half> %res, <64 x half>* %a
  ret void
}

define void @fsub_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: fsub_v128f16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %res = fsub <128 x half> %op1, %op2
  store <128 x half> %res, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @fsub_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: fsub_v2f32:
; CHECK: fsub v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = fsub <2 x float> %op1, %op2
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fsub_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: fsub_v4f32:
; CHECK: fsub v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = fsub <4 x float> %op1, %op2
  ret <4 x float> %res
}

define void @fsub_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: fsub_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = fsub <8 x float> %op1, %op2
  store <8 x float> %res, <8 x float>* %a
  ret void
}

define void @fsub_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: fsub_v16f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = fsub <16 x float> %op1, %op2
  store <16 x float> %res, <16 x float>* %a
  ret void
}

define void @fsub_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: fsub_v32f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = fsub <32 x float> %op1, %op2
  store <32 x float> %res, <32 x float>* %a
  ret void
}

define void @fsub_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: fsub_v64f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %res = fsub <64 x float> %op1, %op2
  store <64 x float> %res, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @fsub_v1f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: fsub_v1f64:
; CHECK: fsub d0, d0, d1
; CHECK: ret
  %res = fsub <1 x double> %op1, %op2
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fsub_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: fsub_v2f64:
; CHECK: fsub v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = fsub <2 x double> %op1, %op2
  ret <2 x double> %res
}

define void @fsub_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: fsub_v4f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = fsub <4 x double> %op1, %op2
  store <4 x double> %res, <4 x double>* %a
  ret void
}

define void @fsub_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: fsub_v8f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = fsub <8 x double> %op1, %op2
  store <8 x double> %res, <8 x double>* %a
  ret void
}

define void @fsub_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: fsub_v16f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = fsub <16 x double> %op1, %op2
  store <16 x double> %res, <16 x double>* %a
  ret void
}

define void @fsub_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: fsub_v32f64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: fsub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %res = fsub <32 x double> %op1, %op2
  store <32 x double> %res, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }

declare <4 x half> @llvm.fma.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <16 x half> @llvm.fma.v16f16(<16 x half>, <16 x half>, <16 x half>)
declare <32 x half> @llvm.fma.v32f16(<32 x half>, <32 x half>, <32 x half>)
declare <64 x half> @llvm.fma.v64f16(<64 x half>, <64 x half>, <64 x half>)
declare <128 x half> @llvm.fma.v128f16(<128 x half>, <128 x half>, <128 x half>)
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)
declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>)
declare <32 x float> @llvm.fma.v32f32(<32 x float>, <32 x float>, <32 x float>)
declare <64 x float> @llvm.fma.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <1 x double> @llvm.fma.v1f64(<1 x double>, <1 x double>, <1 x double>)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
declare <8 x double> @llvm.fma.v8f64(<8 x double>, <8 x double>, <8 x double>)
declare <16 x double> @llvm.fma.v16f64(<16 x double>, <16 x double>, <16 x double>)
declare <32 x double> @llvm.fma.v32f64(<32 x double>, <32 x double>, <32 x double>)

declare <4 x half> @llvm.sqrt.v4f16(<4 x half>)
declare <8 x half> @llvm.sqrt.v8f16(<8 x half>)
declare <16 x half> @llvm.sqrt.v16f16(<16 x half>)
declare <32 x half> @llvm.sqrt.v32f16(<32 x half>)
declare <64 x half> @llvm.sqrt.v64f16(<64 x half>)
declare <128 x half> @llvm.sqrt.v128f16(<128 x half>)
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>)
declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)
declare <32 x float> @llvm.sqrt.v32f32(<32 x float>)
declare <64 x float> @llvm.sqrt.v64f32(<64 x float>)
declare <1 x double> @llvm.sqrt.v1f64(<1 x double>)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <8 x double> @llvm.sqrt.v8f64(<8 x double>)
declare <16 x double> @llvm.sqrt.v16f64(<16 x double>)
declare <32 x double> @llvm.sqrt.v32f64(<32 x double>)
