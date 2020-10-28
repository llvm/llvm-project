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
; AND
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @and_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: and_v8i8:
; CHECK: and v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = and <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @and_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: and_v16i8:
; CHECK: and v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = and <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @and_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: and_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = and <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @and_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: and_v64i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_256-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_256-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_256-DAG: and [[RES_1:z[0-9]+]].d, [[OP1_1]].d, [[OP2_1]].d
; VBITS_LE_256-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = and <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @and_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: and_v128i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_512-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_512-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_512-DAG: and [[RES_1:z[0-9]+]].d, [[OP1_1]].d, [[OP2_1]].d
; VBITS_LE_512-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; VBITS_LE_256-DAG: mov w[[OFF_2:[0-9]+]], #[[#mul(VBYTES,2)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_2:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_2]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_2:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_2]]]
; VBITS_LE_256-DAG: and [[RES_2:z[0-9]+]].d, [[OP1_2]].d, [[OP2_2]].d
; VBITS_LE_256-DAG: st1b { [[RES_2]].b }, [[PG]], [x0, x[[OFF_2]]]
; VBITS_LE_256-DAG: mov w[[OFF_3:[0-9]+]], #[[#mul(VBYTES,3)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_3:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_3]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_3:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_3]]]
; VBITS_LE_256-DAG: and [[RES_3:z[0-9]+]].d, [[OP1_3]].d, [[OP2_3]].d
; VBITS_LE_256-DAG: st1b { [[RES_3]].b }, [[PG]], [x0, x[[OFF_3]]]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = and <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @and_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: and_v256i8:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-DAG: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK-DAG: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_LE_1024-DAG: mov w[[OFF_1:[0-9]+]], #[[#VBYTES]]
; VBITS_LE_1024-DAG: ld1b { [[OP1_1:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_1]]]
; VBITS_LE_1024-DAG: ld1b { [[OP2_1:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_1]]]
; VBITS_LE_1024-DAG: and [[RES_1:z[0-9]+]].d, [[OP1_1]].d, [[OP2_1]].d
; VBITS_LE_1024-DAG: st1b { [[RES_1]].b }, [[PG]], [x0, x[[OFF_1]]]
; VBITS_LE_512-DAG: mov w[[OFF_2:[0-9]+]], #[[#mul(VBYTES,2)]]
; VBITS_LE_512-DAG: ld1b { [[OP1_2:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_2]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_2:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_2]]]
; VBITS_LE_512-DAG: and [[RES_2:z[0-9]+]].d, [[OP1_2]].d, [[OP2_2]].d
; VBITS_LE_512-DAG: st1b { [[RES_2]].b }, [[PG]], [x0, x[[OFF_2]]]
; VBITS_LE_512-DAG: mov w[[OFF_3:[0-9]+]], #[[#mul(VBYTES,3)]]
; VBITS_LE_512-DAG: ld1b { [[OP1_3:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_3]]]
; VBITS_LE_512-DAG: ld1b { [[OP2_3:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_3]]]
; VBITS_LE_512-DAG: and [[RES_3:z[0-9]+]].d, [[OP1_3]].d, [[OP2_3]].d
; VBITS_LE_512-DAG: st1b { [[RES_3]].b }, [[PG]], [x0, x[[OFF_3]]]
; VBITS_LE_256-DAG: mov w[[OFF_4:[0-9]+]], #[[#mul(VBYTES,4)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_4:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_4]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_4:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_4]]]
; VBITS_LE_256-DAG: and [[RES_4:z[0-9]+]].d, [[OP1_4]].d, [[OP2_4]].d
; VBITS_LE_256-DAG: st1b { [[RES_4]].b }, [[PG]], [x0, x[[OFF_4]]]
; VBITS_LE_256-DAG: mov w[[OFF_5:[0-9]+]], #[[#mul(VBYTES,5)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_5:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_5]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_5:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_5]]]
; VBITS_LE_256-DAG: and [[RES_5:z[0-9]+]].d, [[OP1_5]].d, [[OP2_5]].d
; VBITS_LE_256-DAG: st1b { [[RES_5]].b }, [[PG]], [x0, x[[OFF_5]]]
; VBITS_LE_256-DAG: mov w[[OFF_6:[0-9]+]], #[[#mul(VBYTES,6)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_6:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_6]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_6:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_6]]]
; VBITS_LE_256-DAG: and [[RES_6:z[0-9]+]].d, [[OP1_6]].d, [[OP2_6]].d
; VBITS_LE_256-DAG: st1b { [[RES_6]].b }, [[PG]], [x0, x[[OFF_6]]]
; VBITS_LE_256-DAG: mov w[[OFF_7:[0-9]+]], #[[#mul(VBYTES,7)]]
; VBITS_LE_256-DAG: ld1b { [[OP1_7:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_7]]]
; VBITS_LE_256-DAG: ld1b { [[OP2_7:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_7]]]
; VBITS_LE_256-DAG: and [[RES_7:z[0-9]+]].d, [[OP1_7]].d, [[OP2_7]].d
; VBITS_LE_256-DAG: st1b { [[RES_7]].b }, [[PG]], [x0, x[[OFF_7]]]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = and <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @and_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: and_v4i16:
; CHECK: and v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = and <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @and_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: and_v8i16:
; CHECK: and v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = and <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @and_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: and_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = and <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: and_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = and <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: and_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = and <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: and_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = and <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @and_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: and_v2i32:
; CHECK: and v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = and <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @and_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: and_v4i32:
; CHECK: and v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = and <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @and_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: and_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = and <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: and_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = and <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: and_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = and <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: and_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = and <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @and_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: and_v1i64:
; CHECK: and v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = and <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @and_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: and_v2i64:
; CHECK: and v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = and <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @and_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: and_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = and <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: and_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = and <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: and_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = and <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

; NOTE: Check lines only cover the first VBYTES because the and_v#i8 tests
; already cover the general legalisation cases.
define void @and_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: and_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: and [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = and <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; NOTE: Tests beyond this point only have CHECK lines to validate the first
; VBYTES because the and tests already validate the legalisation code paths.
;

;
; OR
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @or_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: or_v8i8:
; CHECK: orr v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = or <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @or_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: or_v16i8:
; CHECK: orr v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = or <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @or_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: or_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = or <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @or_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: or_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = or <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @or_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: or_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = or <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @or_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: or_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = or <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @or_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: or_v4i16:
; CHECK: orr v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = or <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @or_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: or_v8i16:
; CHECK: orr v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = or <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @or_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: or_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = or <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @or_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: or_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = or <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @or_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: or_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = or <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @or_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: or_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = or <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @or_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: or_v2i32:
; CHECK: orr v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = or <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @or_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: or_v4i32:
; CHECK: orr v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = or <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @or_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: or_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = or <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @or_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: or_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = or <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @or_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: or_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = or <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @or_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: or_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = or <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @or_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: or_v1i64:
; CHECK: orr v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = or <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @or_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: or_v2i64:
; CHECK: orr v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = or <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @or_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: or_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = or <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @or_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: or_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = or <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @or_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: or_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = or <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @or_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: or_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: orr [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = or <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; XOR
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @xor_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: xor_v8i8:
; CHECK: eor v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = xor <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @xor_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: xor_v16i8:
; CHECK: eor v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = xor <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @xor_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: xor_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = xor <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @xor_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: xor_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = xor <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @xor_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: xor_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = xor <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @xor_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: xor_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = xor <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @xor_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: xor_v4i16:
; CHECK: eor v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = xor <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @xor_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: xor_v8i16:
; CHECK: eor v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = xor <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @xor_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: xor_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = xor <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @xor_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: xor_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = xor <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @xor_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: xor_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = xor <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @xor_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: xor_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = xor <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @xor_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: xor_v2i32:
; CHECK: eor v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = xor <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @xor_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: xor_v4i32:
; CHECK: eor v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = xor <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @xor_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: xor_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = xor <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @xor_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: xor_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = xor <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @xor_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: xor_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = xor <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @xor_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: xor_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = xor <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @xor_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: xor_v1i64:
; CHECK: eor v0.8b, v0.8b, v1.8b
; CHECK: ret
  %res = xor <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @xor_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: xor_v2i64:
; CHECK: eor v0.16b, v0.16b, v1.16b
; CHECK: ret
  %res = xor <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @xor_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: xor_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = xor <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @xor_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: xor_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = xor <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @xor_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: xor_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = xor <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @xor_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: xor_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK: eor [[RES:z[0-9]+]].d, [[OP1]].d, [[OP2]].d
; CHECK: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = xor <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
