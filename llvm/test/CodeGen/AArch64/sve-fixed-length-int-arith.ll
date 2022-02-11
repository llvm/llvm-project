; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=VBITS_EQ_128
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

;
; ADD
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @add_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: add_v8i8:
; CHECK: add v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = add <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @add_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: add_v16i8:
; CHECK: add v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = add <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @add_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: add_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = add <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @add_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: add_v64i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: add [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_256-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_256-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_256-DAG: add [[RES_1:z[0-9]+]].b, [[PG]]/m, [[OP1_1]].b, [[OP2_1]].b
; VBITS_LE_256-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = add <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @add_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: add_v128i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: add [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_512-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_512-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_512-DAG: add [[RES_1:z[0-9]+]].b, [[PG]]/m, [[OP1_1]].b, [[OP2_1]].b
; VBITS_LE_512-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; VBITS_LE_256-DAG: mov w[[OFF_2:[0-9]+]], #[[#mul(VBYTES,2)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_2:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_2]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_2:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_2]]]
; VBITS_LE_256-DAG: add [[RES_2:z[0-9]+]].b, [[PG]]/m, [[OP1_2]].b, [[OP2_2]].b
; VBITS_LE_256-DAG: st1b { [[RES_2]].b }, [[PG]], [x0, x[[OFF_2]]]
; VBITS_LE_256-DAG: mov w[[OFF_3:[0-9]+]], #[[#mul(VBYTES,3)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_3:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_3]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_3:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_3]]]
; VBITS_LE_256-DAG: add [[RES_3:z[0-9]+]].b, [[PG]]/m, [[OP1_3]].b, [[OP2_3]].b
; VBITS_LE_256-DAG: st1b { [[RES_3]].b }, [[PG]], [x0, x[[OFF_3]]]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = add <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @add_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: add_v256i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: add [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_1024-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_1024-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_1024-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_1024-DAG: add [[RES_1:z[0-9]+]].b, [[PG]]/m, [[OP1_1]].b, [[OP2_1]].b
; VBITS_LE_1024-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; VBITS_LE_512-DAG: mov w[[OFF_2:[0-9]+]], #[[#mul(VBYTES,2)]]
; VBITS_LE_512-DAG: ld1b { [[OP1_2:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_2]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_2:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_2]]]
; VBITS_LE_512-DAG: add [[RES_2:z[0-9]+]].b, [[PG]]/m, [[OP1_2]].b, [[OP2_2]].b
; VBITS_LE_512-DAG: st1b { [[RES_2]].b }, [[PG]], [x0, x[[OFF_2]]]
; VBITS_LE_512-DAG: mov w[[OFF_3:[0-9]+]], #[[#mul(VBYTES,3)]]
; VBITS_LE_512-DAG: ld1b { [[OP1_3:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_3]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_3:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_3]]]
; VBITS_LE_512-DAG: add [[RES_3:z[0-9]+]].b, [[PG]]/m, [[OP1_3]].b, [[OP2_3]].b
; VBITS_LE_512-DAG: st1b { [[RES_3]].b }, [[PG]], [x0, x[[OFF_3]]]
; VBITS_LE_256-DAG: mov w[[OFF_4:[0-9]+]], #[[#mul(VBYTES,4)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_4:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_4]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_4:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_4]]]
; VBITS_LE_256-DAG: add [[RES_4:z[0-9]+]].b, [[PG]]/m, [[OP1_4]].b, [[OP2_4]].b
; VBITS_LE_256-DAG: st1b { [[RES_4]].b }, [[PG]], [x0, x[[OFF_4]]]
; VBITS_LE_256-DAG: mov w[[OFF_5:[0-9]+]], #[[#mul(VBYTES,5)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_5:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_5]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_5:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_5]]]
; VBITS_LE_256-DAG: add [[RES_5:z[0-9]+]].b, [[PG]]/m, [[OP1_5]].b, [[OP2_5]].b
; VBITS_LE_256-DAG: st1b { [[RES_5]].b }, [[PG]], [x0, x[[OFF_5]]]
; VBITS_LE_256-DAG: mov w[[OFF_6:[0-9]+]], #[[#mul(VBYTES,6)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_6:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_6]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_6:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_6]]]
; VBITS_LE_256-DAG: add [[RES_6:z[0-9]+]].b, [[PG]]/m, [[OP1_6]].b, [[OP2_6]].b
; VBITS_LE_256-DAG: st1b { [[RES_6]].b }, [[PG]], [x0, x[[OFF_6]]]
; VBITS_LE_256-DAG: mov w[[OFF_7:[0-9]+]], #[[#mul(VBYTES,7)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_7:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_7]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_7:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_7]]]
; VBITS_LE_256-DAG: add [[RES_7:z[0-9]+]].b, [[PG]]/m, [[OP1_7]].b, [[OP2_7]].b
; VBITS_LE_256-DAG: st1b { [[RES_7]].b }, [[PG]], [x0, x[[OFF_7]]]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = add <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @add_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: add_v4i16:
; CHECK: add v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = add <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @add_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: add_v8i16:
; CHECK: add v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = add <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @add_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: add_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = add <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: add_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = add <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: add_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = add <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: add_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = add <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @add_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: add_v2i32:
; CHECK: add v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = add <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @add_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: add_v4i32:
; CHECK: add v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = add <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @add_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: add_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = add <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: add_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = add <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: add_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = add <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: add_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = add <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @add_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: add_v1i64:
; CHECK: add d0, d0, d1
; CHECK: ret
  %res = add <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @add_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: add_v2i64:
; CHECK: add v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = add <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @add_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: add_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = add <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: add_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = add <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: add_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = add <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the add_v#i8 tests
; already cover the general legalisation cases.
define void @add_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: add_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: add [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = add <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; NOTE: Tests beyond this point only have CHECK lines to validate the first
; VBYTES because the add tests already validate the legalisation code paths.
;

;
; MUL
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @mul_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: mul_v8i8:
; CHECK: mul v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = mul <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @mul_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: mul_v16i8:
; CHECK: mul v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = mul <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @mul_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: mul_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = mul <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @mul_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: mul_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = mul <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @mul_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: mul_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = mul <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @mul_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: mul_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = mul <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @mul_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: mul_v4i16:
; CHECK: mul v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = mul <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @mul_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: mul_v8i16:
; CHECK: mul v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = mul <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @mul_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: mul_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = mul <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @mul_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: mul_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = mul <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @mul_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: mul_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = mul <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @mul_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: mul_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = mul <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @mul_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: mul_v2i32:
; CHECK: mul v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = mul <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @mul_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: mul_v4i32:
; CHECK: mul v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = mul <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @mul_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: mul_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = mul <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @mul_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: mul_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = mul <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @mul_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: mul_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = mul <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @mul_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: mul_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = mul <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

define <1 x i64> @mul_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: mul_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK: mul z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret

; VBITS_EQ_128-LABEL: mul_v1i64:
; VBITS_EQ_128:         ptrue p0.d, vl1
; VBITS_EQ_128:         mul z0.d, p0/m, z0.d, z1.d
; VBITS_EQ_128:         ret

  %res = mul <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

define <2 x i64> @mul_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: mul_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: mul z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret

; VBITS_EQ_128-LABEL: mul_v2i64:
; VBITS_EQ_128:         ptrue p0.d, vl2
; VBITS_EQ_128:         mul z0.d, p0/m, z0.d, z1.d
; VBITS_EQ_128:         ret

  %res = mul <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @mul_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: mul_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = mul <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @mul_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: mul_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = mul <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @mul_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: mul_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = mul <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @mul_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: mul_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: mul [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = mul <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; SUB
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @sub_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: sub_v8i8:
; CHECK: sub v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = sub <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @sub_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: sub_v16i8:
; CHECK: sub v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = sub <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @sub_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: sub_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = sub <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @sub_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: sub_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = sub <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @sub_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: sub_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = sub <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @sub_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: sub_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[OP2]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = sub <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @sub_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: sub_v4i16:
; CHECK: sub v0.4h, v0.4h, v1.4h
; CHECK: ret
  %res = sub <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @sub_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: sub_v8i16:
; CHECK: sub v0.8h, v0.8h, v1.8h
; CHECK: ret
  %res = sub <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @sub_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: sub_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = sub <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @sub_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: sub_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = sub <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @sub_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: sub_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = sub <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @sub_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: sub_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[OP2]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = sub <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @sub_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: sub_v2i32:
; CHECK: sub v0.2s, v0.2s, v1.2s
; CHECK: ret
  %res = sub <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @sub_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: sub_v4i32:
; CHECK: sub v0.4s, v0.4s, v1.4s
; CHECK: ret
  %res = sub <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @sub_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: sub_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = sub <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @sub_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: sub_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = sub <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @sub_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: sub_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = sub <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @sub_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: sub_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = sub <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @sub_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: sub_v1i64:
; CHECK: sub d0, d0, d1
; CHECK: ret
  %res = sub <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @sub_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: sub_v2i64:
; CHECK: sub v0.2d, v0.2d, v1.2d
; CHECK: ret
  %res = sub <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @sub_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: sub_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = sub <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @sub_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: sub_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = sub <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @sub_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: sub_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = sub <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @sub_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: sub_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = sub <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}


;
; ABS
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @abs_v8i8(<8 x i8> %op1) #0 {
; CHECK-LABEL: abs_v8i8:
; CHECK: abs v0.8b, v0.8b
; CHECK: ret
  %res = call <8 x i8> @llvm.abs.v8i8(<8 x i8> %op1, i1 false)
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @abs_v16i8(<16 x i8> %op1) #0 {
; CHECK-LABEL: abs_v16i8:
; CHECK: abs v0.16b, v0.16b
; CHECK: ret
  %res = call <16 x i8> @llvm.abs.v16i8(<16 x i8> %op1, i1 false)
  ret <16 x i8> %res
}

define void @abs_v32i8(<32 x i8>* %a) #0 {
; CHECK-LABEL: abs_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %res = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %op1, i1 false)
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @abs_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: abs_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %res = call <64 x i8> @llvm.abs.v64i8(<64 x i8> %op1, i1 false)
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @abs_v128i8(<128 x i8>* %a) #0 {
; CHECK-LABEL: abs_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %res = call <128 x i8> @llvm.abs.v128i8(<128 x i8> %op1, i1 false)
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @abs_v256i8(<256 x i8>* %a) #0 {
; CHECK-LABEL: abs_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %res = call <256 x i8> @llvm.abs.v256i8(<256 x i8> %op1, i1 false)
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @abs_v4i16(<4 x i16> %op1) #0 {
; CHECK-LABEL: abs_v4i16:
; CHECK: abs v0.4h, v0.4h
; CHECK: ret
  %res = call <4 x i16> @llvm.abs.v4i16(<4 x i16> %op1, i1 false)
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @abs_v8i16(<8 x i16> %op1) #0 {
; CHECK-LABEL: abs_v8i16:
; CHECK: abs v0.8h, v0.8h
; CHECK: ret
  %res = call <8 x i16> @llvm.abs.v8i16(<8 x i16> %op1, i1 false)
  ret <8 x i16> %res
}

define void @abs_v16i16(<16 x i16>* %a) #0 {
; CHECK-LABEL: abs_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %res = call <16 x i16> @llvm.abs.v16i16(<16 x i16> %op1, i1 false)
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @abs_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: abs_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %res = call <32 x i16> @llvm.abs.v32i16(<32 x i16> %op1, i1 false)
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @abs_v64i16(<64 x i16>* %a) #0 {
; CHECK-LABEL: abs_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %res = call <64 x i16> @llvm.abs.v64i16(<64 x i16> %op1, i1 false)
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @abs_v128i16(<128 x i16>* %a) #0 {
; CHECK-LABEL: abs_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %res = call <128 x i16> @llvm.abs.v128i16(<128 x i16> %op1, i1 false)
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @abs_v2i32(<2 x i32> %op1) #0 {
; CHECK-LABEL: abs_v2i32:
; CHECK: abs v0.2s, v0.2s
; CHECK: ret
  %res = call <2 x i32> @llvm.abs.v2i32(<2 x i32> %op1, i1 false)
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @abs_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: abs_v4i32:
; CHECK: abs v0.4s, v0.4s
; CHECK: ret
  %res = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %op1, i1 false)
  ret <4 x i32> %res
}

define void @abs_v8i32(<8 x i32>* %a) #0 {
; CHECK-LABEL: abs_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = call <8 x i32> @llvm.abs.v8i32(<8 x i32> %op1, i1 false)
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @abs_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: abs_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %res = call <16 x i32> @llvm.abs.v16i32(<16 x i32> %op1, i1 false)
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @abs_v32i32(<32 x i32>* %a) #0 {
; CHECK-LABEL: abs_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %res = call <32 x i32> @llvm.abs.v32i32(<32 x i32> %op1, i1 false)
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @abs_v64i32(<64 x i32>* %a) #0 {
; CHECK-LABEL: abs_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %res = call <64 x i32> @llvm.abs.v64i32(<64 x i32> %op1, i1 false)
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @abs_v1i64(<1 x i64> %op1) #0 {
; CHECK-LABEL: abs_v1i64:
; CHECK: abs d0, d0
; CHECK: ret
  %res = call <1 x i64> @llvm.abs.v1i64(<1 x i64> %op1, i1 false)
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @abs_v2i64(<2 x i64> %op1) #0 {
; CHECK-LABEL: abs_v2i64:
; CHECK: abs v0.2d, v0.2d
; CHECK: ret
  %res = call <2 x i64> @llvm.abs.v2i64(<2 x i64> %op1, i1 false)
  ret <2 x i64> %res
}

define void @abs_v4i64(<4 x i64>* %a) #0 {
; CHECK-LABEL: abs_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %res = call <4 x i64> @llvm.abs.v4i64(<4 x i64> %op1, i1 false)
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @abs_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: abs_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %res = call <8 x i64> @llvm.abs.v8i64(<8 x i64> %op1, i1 false)
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @abs_v16i64(<16 x i64>* %a) #0 {
; CHECK-LABEL: abs_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %res = call <16 x i64> @llvm.abs.v16i64(<16 x i64> %op1, i1 false)
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @abs_v32i64(<32 x i64>* %a) #0 {
; CHECK-LABEL: abs_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK: abs [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %res = call <32 x i64> @llvm.abs.v32i64(<32 x i64> %op1, i1 false)
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

declare <8 x i8> @llvm.abs.v8i8(<8 x i8>, i1)
declare <16 x i8> @llvm.abs.v16i8(<16 x i8>, i1)
declare <32 x i8> @llvm.abs.v32i8(<32 x i8>, i1)
declare <64 x i8> @llvm.abs.v64i8(<64 x i8>, i1)
declare <128 x i8> @llvm.abs.v128i8(<128 x i8>, i1)
declare <256 x i8> @llvm.abs.v256i8(<256 x i8>, i1)
declare <4 x i16> @llvm.abs.v4i16(<4 x i16>, i1)
declare <8 x i16> @llvm.abs.v8i16(<8 x i16>, i1)
declare <16 x i16> @llvm.abs.v16i16(<16 x i16>, i1)
declare <32 x i16> @llvm.abs.v32i16(<32 x i16>, i1)
declare <64 x i16> @llvm.abs.v64i16(<64 x i16>, i1)
declare <128 x i16> @llvm.abs.v128i16(<128 x i16>, i1)
declare <2 x i32> @llvm.abs.v2i32(<2 x i32>, i1)
declare <4 x i32> @llvm.abs.v4i32(<4 x i32>, i1)
declare <8 x i32> @llvm.abs.v8i32(<8 x i32>, i1)
declare <16 x i32> @llvm.abs.v16i32(<16 x i32>, i1)
declare <32 x i32> @llvm.abs.v32i32(<32 x i32>, i1)
declare <64 x i32> @llvm.abs.v64i32(<64 x i32>, i1)
declare <1 x i64> @llvm.abs.v1i64(<1 x i64>, i1)
declare <2 x i64> @llvm.abs.v2i64(<2 x i64>, i1)
declare <4 x i64> @llvm.abs.v4i64(<4 x i64>, i1)
declare <8 x i64> @llvm.abs.v8i64(<8 x i64>, i1)
declare <16 x i64> @llvm.abs.v16i64(<16 x i64>, i1)
declare <32 x i64> @llvm.abs.v32i64(<32 x i64>, i1)

attributes #0 = { "target-features"="+sve" }
