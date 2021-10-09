; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=16 -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=64 -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

; Don't use SVE for 64-bit vectors.
define <8 x i8> @select_v8i8(<8 x i8> %op1, <8 x i8> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v8i8:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.8b, w8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <8 x i8> %op1, <8 x i8> %op2
  ret <8 x i8> %sel
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @select_v16i8(<16 x i8> %op1, <16 x i8> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v16i8:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.16b, w8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <16 x i8> %op1, <16 x i8> %op2
  ret <16 x i8> %sel
}

define void @select_v32i8(<32 x i8>* %a, <32 x i8>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v32i8:
; CHECK: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].b
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].b, w[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].b, [[TMP1]].b, #0x1
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].b, [[PG2]]/z, [[TMP2]].b, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].b, [[PRES]], [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <32 x i8>, <32 x i8>* %a
  %op2 = load volatile <32 x i8>, <32 x i8>* %b
  %sel = select i1 %mask, <32 x i8> %op1, <32 x i8> %op2
  store <32 x i8> %sel, <32 x i8>* %a
  ret void
}

define void @select_v64i8(<64 x i8>* %a, <64 x i8>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v64i8:
; VBITS_GE_512: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; VBITS_GE_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].b
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].b, w[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].b, [[TMP1]].b, #0x1
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].b, [[PG2]]/z, [[TMP2]].b, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].b, [[PRES]], [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <64 x i8>, <64 x i8>* %a
  %op2 = load volatile <64 x i8>, <64 x i8>* %b
  %sel = select i1 %mask, <64 x i8> %op1, <64 x i8> %op2
  store <64 x i8> %sel, <64 x i8>* %a
  ret void
}

define void @select_v128i8(<128 x i8>* %a, <128 x i8>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v128i8:
; VBITS_GE_1024: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].b
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].b, w[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].b, [[TMP1]].b, #0x1
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].b, [[PG2]]/z, [[TMP2]].b, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].b, [[PRES]], [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <128 x i8>, <128 x i8>* %a
  %op2 = load volatile <128 x i8>, <128 x i8>* %b
  %sel = select i1 %mask, <128 x i8> %op1, <128 x i8> %op2
  store <128 x i8> %sel, <128 x i8>* %a
  ret void
}

define void @select_v256i8(<256 x i8>* %a, <256 x i8>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v256i8:
; VBITS_GE_2048: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].b
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].b, w[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].b, [[TMP1]].b, #0x1
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].b, [[PG2]]/z, [[TMP2]].b, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].b, [[PRES]], [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <256 x i8>, <256 x i8>* %a
  %op2 = load volatile <256 x i8>, <256 x i8>* %b
  %sel = select i1 %mask, <256 x i8> %op1, <256 x i8> %op2
  store <256 x i8> %sel, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @select_v4i16(<4 x i16> %op1, <4 x i16> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v4i16:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.4h, w8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <4 x i16> %op1, <4 x i16> %op2
  ret <4 x i16> %sel
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @select_v8i16(<8 x i16> %op1, <8 x i16> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v8i16:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.8h, w8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <8 x i16> %op1, <8 x i16> %op2
  ret <8 x i16> %sel
}

define void @select_v16i16(<16 x i16>* %a, <16 x i16>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v16i16:
; CHECK: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].h
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <16 x i16>, <16 x i16>* %a
  %op2 = load volatile <16 x i16>, <16 x i16>* %b
  %sel = select i1 %mask, <16 x i16> %op1, <16 x i16> %op2
  store <16 x i16> %sel, <16 x i16>* %a
  ret void
}

define void @select_v32i16(<32 x i16>* %a, <32 x i16>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v32i16:
; VBITS_GE_512: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <32 x i16>, <32 x i16>* %a
  %op2 = load volatile <32 x i16>, <32 x i16>* %b
  %sel = select i1 %mask, <32 x i16> %op1, <32 x i16> %op2
  store <32 x i16> %sel, <32 x i16>* %a
  ret void
}

define void @select_v64i16(<64 x i16>* %a, <64 x i16>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v64i16:
; VBITS_GE_1024: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <64 x i16>, <64 x i16>* %a
  %op2 = load volatile <64 x i16>, <64 x i16>* %b
  %sel = select i1 %mask, <64 x i16> %op1, <64 x i16> %op2
  store <64 x i16> %sel, <64 x i16>* %a
  ret void
}

define void @select_v128i16(<128 x i16>* %a, <128 x i16>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v128i16:
; VBITS_GE_2048: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <128 x i16>, <128 x i16>* %a
  %op2 = load volatile <128 x i16>, <128 x i16>* %b
  %sel = select i1 %mask, <128 x i16> %op1, <128 x i16> %op2
  store <128 x i16> %sel, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @select_v2i32(<2 x i32> %op1, <2 x i32> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v2i32:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.2s, w8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <2 x i32> %op1, <2 x i32> %op2
  ret <2 x i32> %sel
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @select_v4i32(<4 x i32> %op1, <4 x i32> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v4i32:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.4s, w8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <4 x i32> %op1, <4 x i32> %op2
  ret <4 x i32> %sel
}

define void @select_v8i32(<8 x i32>* %a, <8 x i32>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v8i32:
; CHECK: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <8 x i32>, <8 x i32>* %a
  %op2 = load volatile <8 x i32>, <8 x i32>* %b
  %sel = select i1 %mask, <8 x i32> %op1, <8 x i32> %op2
  store <8 x i32> %sel, <8 x i32>* %a
  ret void
}

define void @select_v16i32(<16 x i32>* %a, <16 x i32>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v16i32:
; VBITS_GE_512: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <16 x i32>, <16 x i32>* %a
  %op2 = load volatile <16 x i32>, <16 x i32>* %b
  %sel = select i1 %mask, <16 x i32> %op1, <16 x i32> %op2
  store <16 x i32> %sel, <16 x i32>* %a
  ret void
}

define void @select_v32i32(<32 x i32>* %a, <32 x i32>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v32i32:
; VBITS_GE_1024: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <32 x i32>, <32 x i32>* %a
  %op2 = load volatile <32 x i32>, <32 x i32>* %b
  %sel = select i1 %mask, <32 x i32> %op1, <32 x i32> %op2
  store <32 x i32> %sel, <32 x i32>* %a
  ret void
}

define void @select_v64i32(<64 x i32>* %a, <64 x i32>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v64i32:
; VBITS_GE_2048: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <64 x i32>, <64 x i32>* %a
  %op2 = load volatile <64 x i32>, <64 x i32>* %b
  %sel = select i1 %mask, <64 x i32> %op1, <64 x i32> %op2
  store <64 x i32> %sel, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @select_v1i64(<1 x i64> %op1, <1 x i64> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v1i64:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm x8, ne
; CHECK-NEXT: fmov d2, x8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <1 x i64> %op1, <1 x i64> %op2
  ret <1 x i64> %sel
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @select_v2i64(<2 x i64> %op1, <2 x i64> %op2, i1 %mask) #0 {
; CHECK-LABEL: select_v2i64:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm x8, ne
; CHECK-NEXT: dup v2.2d, x8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <2 x i64> %op1, <2 x i64> %op2
  ret <2 x i64> %sel
}

define void @select_v4i64(<4 x i64>* %a, <4 x i64>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v4i64:
; CHECK: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <4 x i64>, <4 x i64>* %a
  %op2 = load volatile <4 x i64>, <4 x i64>* %b
  %sel = select i1 %mask, <4 x i64> %op1, <4 x i64> %op2
  store <4 x i64> %sel, <4 x i64>* %a
  ret void
}

define void @select_v8i64(<8 x i64>* %a, <8 x i64>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v8i64:
; VBITS_GE_512: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <8 x i64>, <8 x i64>* %a
  %op2 = load volatile <8 x i64>, <8 x i64>* %b
  %sel = select i1 %mask, <8 x i64> %op1, <8 x i64> %op2
  store <8 x i64> %sel, <8 x i64>* %a
  ret void
}

define void @select_v16i64(<16 x i64>* %a, <16 x i64>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v16i64:
; VBITS_GE_1024: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <16 x i64>, <16 x i64>* %a
  %op2 = load volatile <16 x i64>, <16 x i64>* %b
  %sel = select i1 %mask, <16 x i64> %op1, <16 x i64> %op2
  store <16 x i64> %sel, <16 x i64>* %a
  ret void
}

define void @select_v32i64(<32 x i64>* %a, <32 x i64>* %b, i1 %mask) #0 {
; CHECK-LABEL: select_v32i64:
; VBITS_GE_2048: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <32 x i64>, <32 x i64>* %a
  %op2 = load volatile <32 x i64>, <32 x i64>* %b
  %sel = select i1 %mask, <32 x i64> %op1, <32 x i64> %op2
  store <32 x i64> %sel, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
