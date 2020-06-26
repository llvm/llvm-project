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

; Don't use SVE for 64-bit vectors.
define <4 x half> @fadd_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: @fadd_v4f16
; CHECK: fadd v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = fadd <4 x half> %op1, %op2
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @fadd_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: @fadd_v8f16
; CHECK: fadd v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = fadd <8 x half> %op1, %op2
  ret <8 x half> %res
}

define void @fadd_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: @fadd_v16f16
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
; CHECK-LABEL: @fadd_v32f16
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
; CHECK-LABEL: @fadd_v64f16
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; CHECK-LABEL: @fadd_v128f16
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
; CHECK-LABEL: @fadd_v2f32
; CHECK: fadd v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = fadd <2 x float> %op1, %op2
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @fadd_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: @fadd_v4f32
; CHECK: fadd v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = fadd <4 x float> %op1, %op2
  ret <4 x float> %res
}

define void @fadd_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: @fadd_v8f32
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; CHECK-LABEL: @fadd_v16f32
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; CHECK-LABEL: @fadd_v32f32
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; CHECK-LABEL: @fadd_v64f32
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
; CHECK-LABEL: @fadd_v1f64
; CHECK: fadd d0, d0, d1
; CHECK: ret
  %res = fadd <1 x double> %op1, %op2
  ret <1 x double> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @fadd_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: @fadd_v2f64
; CHECK: fadd v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = fadd <2 x double> %op1, %op2
  ret <2 x double> %res
}

define void @fadd_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: @fadd_v4f64
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; CHECK-LABEL: @fadd_v8f64
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; CHECK-LABEL: @fadd_v16f64
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

; NOTE: Check lines only cover the first VBYTES because the add_v#f16 tests
; already cover the general legalisation cases.
define void @fadd_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; CHECK-LABEL: @fadd_v32f64
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

attributes #0 = { "target-features"="+sve" }
