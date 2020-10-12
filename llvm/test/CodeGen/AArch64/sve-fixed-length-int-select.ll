; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=16 -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32 -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -D#VBYTES=32 -check-prefixes=CHECK
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
define <8 x i8> @select_v8i8(<8 x i8> %op1, <8 x i8> %op2, <8 x i1> %mask) #0 {
; CHECK: select_v8i8:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <8 x i1> %mask, <8 x i8> %op1, <8 x i8> %op2
  ret <8 x i8> %sel
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @select_v16i8(<16 x i8> %op1, <16 x i8> %op2, <16 x i1> %mask) #0 {
; CHECK: select_v16i8:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <16 x i1> %mask, <16 x i8> %op1, <16 x i8> %op2
  ret <16 x i8> %sel
}

define void @select_v32i8(<32 x i8>* %a, <32 x i8>* %b, <32 x i1>* %c) #0 {
; CHECK: select_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,32)]]
; CHECK: ptrue [[PG1:p[0-9]+]].b
; CHECK: ld1b { [[MASK:z[0-9]+]].b }, [[PG]]/z, [x9]
; CHECK-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: and [[AND:z[0-9]+]].b, [[MASK]].b, #0x1
; CHECK-NEXT: cmpne [[COND:p[0-9]+]].b, [[PG1]]/z, [[AND]].b, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].b, [[COND]], [[OP1]].b, [[OP2]].b
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; CHECK: ret
  %mask = load <32 x i1>, <32 x i1>* %c
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %sel = select <32 x i1> %mask, <32 x i8> %op1, <32 x i8> %op2
  store <32 x i8> %sel, <32 x i8>* %a
  ret void
}

define void @select_v64i8(<64 x i8>* %a, <64 x i8>* %b, <64 x i1>* %c) #0 {
; CHECK: select_v64i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,64)]]
; CHECK: ptrue [[PG1:p[0-9]+]].b
; VBITS_GE_512: ld1b { [[MASK:z[0-9]+]].b }, [[PG]]/z, [x9]
; VBITS_GE_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: and [[AND:z[0-9]+]].b, [[MASK]].b, #0x1
; VBITS_GE_512-NEXT: cmpne [[COND:p[0-9]+]].b, [[PG1]]/z, [[AND]].b, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].b, [[COND]], [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512: ret
  %mask = load <64 x i1>, <64 x i1>* %c
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %sel = select <64 x i1> %mask, <64 x i8> %op1, <64 x i8> %op2
  store <64 x i8> %sel, <64 x i8>* %a
  ret void
}

define void @select_v128i8(<128 x i8>* %a, <128 x i8>* %b, <128 x i1>* %c) #0 {
; CHECK: select_v128i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,128)]]
; CHECK: ptrue [[PG1:p[0-9]+]].b
; VBITS_GE_1024: ld1b { [[MASK:z[0-9]+]].b }, [[PG]]/z, [x9]
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: and [[AND:z[0-9]+]].b, [[MASK]].b, #0x1
; VBITS_GE_1024-NEXT: cmpne [[COND:p[0-9]+]].b, [[PG1]]/z, [[AND]].b, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].b, [[COND]], [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %mask = load <128 x i1>, <128 x i1>* %c
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %sel = select <128 x i1> %mask, <128 x i8> %op1, <128 x i8> %op2
  store <128 x i8> %sel, <128 x i8>* %a
  ret void
}

define void @select_v256i8(<256 x i8>* %a, <256 x i8>* %b, <256 x i1>* %c) #0 {
; CHECK: select_v256i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl[[#min(VBYTES,256)]]
; CHECK: ptrue [[PG1:p[0-9]+]].b
; VBITS_GE_2048: ld1b { [[MASK:z[0-9]+]].b }, [[PG]]/z, [x9]
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: and [[AND:z[0-9]+]].b, [[MASK]].b, #0x1
; VBITS_GE_2048-NEXT: cmpne [[COND:p[0-9]+]].b, [[PG1]]/z, [[AND]].b, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].b, [[COND]], [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %mask = load <256 x i1>, <256 x i1>* %c
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %sel = select <256 x i1> %mask, <256 x i8> %op1, <256 x i8> %op2
  store <256 x i8> %sel, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @select_v4i16(<4 x i16> %op1, <4 x i16> %op2, <4 x i1> %mask) #0 {
; CHECK: select_v4i16:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <4 x i1> %mask, <4 x i16> %op1, <4 x i16> %op2
  ret <4 x i16> %sel
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @select_v8i16(<8 x i16> %op1, <8 x i16> %op2, <8 x i1> %mask) #0 {
; CHECK: select_v8i16:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <8 x i1> %mask, <8 x i16> %op1, <8 x i16> %op2
  ret <8 x i16> %sel
}

define void @select_v16i16(<16 x i16>* %a, <16 x i16>* %b, <16 x i1>* %c) #0 {
; CHECK: select_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK: ptrue [[PG1:p[0-9]+]].h
; CHECK: ld1h { [[MASK:z[0-9]+]].h }, [[PG]]/z, [x9]
; CHECK-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: and [[AND:z[0-9]+]].h, [[MASK]].h, #0x1
; CHECK-NEXT: cmpne [[COND:p[0-9]+]].h, [[PG1]]/z, [[AND]].h, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].h, [[COND]], [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK: ret
  %mask = load <16 x i1>, <16 x i1>* %c
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %sel = select <16 x i1> %mask, <16 x i16> %op1, <16 x i16> %op2
  store <16 x i16> %sel, <16 x i16>* %a
  ret void
}

define void @select_v32i16(<32 x i16>* %a, <32 x i16>* %b, <32 x i1>* %c) #0 {
; CHECK: select_v32i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; CHECK: ptrue [[PG1:p[0-9]+]].h
; VBITS_GE_512: ld1h { [[MASK:z[0-9]+]].h }, [[PG]]/z, [x9]
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: and [[AND:z[0-9]+]].h, [[MASK]].h, #0x1
; VBITS_GE_512-NEXT: cmpne [[COND:p[0-9]+]].h, [[PG1]]/z, [[AND]].h, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].h, [[COND]], [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512: ret
  %mask = load <32 x i1>, <32 x i1>* %c
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %sel = select <32 x i1> %mask, <32 x i16> %op1, <32 x i16> %op2
  store <32 x i16> %sel, <32 x i16>* %a
  ret void
}

define void @select_v64i16(<64 x i16>* %a, <64 x i16>* %b, <64 x i1>* %c) #0 {
; CHECK: select_v64i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; CHECK: ptrue [[PG1:p[0-9]+]].h
; VBITS_GE_1024: ld1h { [[MASK:z[0-9]+]].h }, [[PG]]/z, [x9]
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: and [[AND:z[0-9]+]].h, [[MASK]].h, #0x1
; VBITS_GE_1024-NEXT: cmpne [[COND:p[0-9]+]].h, [[PG1]]/z, [[AND]].h, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].h, [[COND]], [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %mask = load <64 x i1>, <64 x i1>* %c
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %sel = select <64 x i1> %mask, <64 x i16> %op1, <64 x i16> %op2
  store <64 x i16> %sel, <64 x i16>* %a
  ret void
}

define void @select_v128i16(<128 x i16>* %a, <128 x i16>* %b, <128 x i1>* %c) #0 {
; CHECK: select_v128i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; CHECK: ptrue [[PG1:p[0-9]+]].h
; VBITS_GE_2048: ld1h { [[MASK:z[0-9]+]].h }, [[PG]]/z, [x9]
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: and [[AND:z[0-9]+]].h, [[MASK]].h, #0x1
; VBITS_GE_2048-NEXT: cmpne [[COND:p[0-9]+]].h, [[PG1]]/z, [[AND]].h, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].h, [[COND]], [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %mask = load <128 x i1>, <128 x i1>* %c
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %sel = select <128 x i1> %mask, <128 x i16> %op1, <128 x i16> %op2
  store <128 x i16> %sel, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @select_v2i32(<2 x i32> %op1, <2 x i32> %op2, <2 x i1> %mask) #0 {
; CHECK: select_v2i32:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <2 x i1> %mask, <2 x i32> %op1, <2 x i32> %op2
  ret <2 x i32> %sel
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @select_v4i32(<4 x i32> %op1, <4 x i32> %op2, <4 x i1> %mask) #0 {
; CHECK: select_v4i32:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <4 x i1> %mask, <4 x i32> %op1, <4 x i32> %op2
  ret <4 x i32> %sel
}

define void @select_v8i32(<8 x i32>* %a, <8 x i32>* %b, <8 x i1>* %c) #0 {
; CHECK: select_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK: ptrue [[PG1:p[0-9]+]].s
; CHECK: ld1w { [[MASK:z[0-9]+]].s }, [[PG]]/z, [x9]
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: and [[AND:z[0-9]+]].s, [[MASK]].s, #0x1
; CHECK-NEXT: cmpne [[COND:p[0-9]+]].s, [[PG1]]/z, [[AND]].s, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].s, [[COND]], [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK: ret
  %mask = load <8 x i1>, <8 x i1>* %c
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %sel = select <8 x i1> %mask, <8 x i32> %op1, <8 x i32> %op2
  store <8 x i32> %sel, <8 x i32>* %a
  ret void
}

define void @select_v16i32(<16 x i32>* %a, <16 x i32>* %b, <16 x i1>* %c) #0 {
; CHECK: select_v16i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK: ptrue [[PG1:p[0-9]+]].s
; VBITS_GE_512: ld1w { [[MASK:z[0-9]+]].s }, [[PG]]/z, [x9]
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: and [[AND:z[0-9]+]].s, [[MASK]].s, #0x1
; VBITS_GE_512-NEXT: cmpne [[COND:p[0-9]+]].s, [[PG1]]/z, [[AND]].s, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].s, [[COND]], [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512: ret
  %mask = load <16 x i1>, <16 x i1>* %c
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %sel = select <16 x i1> %mask, <16 x i32> %op1, <16 x i32> %op2
  store <16 x i32> %sel, <16 x i32>* %a
  ret void
}

define void @select_v32i32(<32 x i32>* %a, <32 x i32>* %b, <32 x i1>* %c) #0 {
; CHECK: select_v32i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK: ptrue [[PG1:p[0-9]+]].s
; VBITS_GE_1024: ld1w { [[MASK:z[0-9]+]].s }, [[PG]]/z, [x9]
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: and [[AND:z[0-9]+]].s, [[MASK]].s, #0x1
; VBITS_GE_1024-NEXT: cmpne [[COND:p[0-9]+]].s, [[PG1]]/z, [[AND]].s, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].s, [[COND]], [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %mask = load <32 x i1>, <32 x i1>* %c
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %sel = select <32 x i1> %mask, <32 x i32> %op1, <32 x i32> %op2
  store <32 x i32> %sel, <32 x i32>* %a
  ret void
}

define void @select_v64i32(<64 x i32>* %a, <64 x i32>* %b, <64 x i1>* %c) #0 {
; CHECK: select_v64i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK: ptrue [[PG1:p[0-9]+]].s
; VBITS_GE_2048: ld1w { [[MASK:z[0-9]+]].s }, [[PG]]/z, [x9]
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: and [[AND:z[0-9]+]].s, [[MASK]].s, #0x1
; VBITS_GE_2048-NEXT: cmpne [[COND:p[0-9]+]].s, [[PG1]]/z, [[AND]].s, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].s, [[COND]], [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %mask = load <64 x i1>, <64 x i1>* %c
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %sel = select <64 x i1> %mask, <64 x i32> %op1, <64 x i32> %op2
  store <64 x i32> %sel, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @select_v1i64(<1 x i64> %op1, <1 x i64> %op2, <1 x i1> %mask) #0 {
; CHECK: select_v1i64:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <1 x i1> %mask, <1 x i64> %op1, <1 x i64> %op2
  ret <1 x i64> %sel
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @select_v2i64(<2 x i64> %op1, <2 x i64> %op2, <2 x i1> %mask) #0 {
; CHECK: select_v2i64:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <2 x i1> %mask, <2 x i64> %op1, <2 x i64> %op2
  ret <2 x i64> %sel
}

define void @select_v4i64(<4 x i64>* %a, <4 x i64>* %b, <4 x i1>* %c) #0 {
; CHECK: select_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK: ptrue [[PG1:p[0-9]+]].d
; CHECK: ld1d { [[MASK:z[0-9]+]].d }, [[PG]]/z, [x9]
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: and [[AND:z[0-9]+]].d, [[MASK]].d, #0x1
; CHECK-NEXT: cmpne [[COND:p[0-9]+]].d, [[PG1]]/z, [[AND]].d, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].d, [[COND]], [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK: ret
  %mask = load <4 x i1>, <4 x i1>* %c
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %sel = select <4 x i1> %mask, <4 x i64> %op1, <4 x i64> %op2
  store <4 x i64> %sel, <4 x i64>* %a
  ret void
}

define void @select_v8i64(<8 x i64>* %a, <8 x i64>* %b, <8 x i1>* %c) #0 {
; CHECK: select_v8i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; CHECK: ptrue [[PG1:p[0-9]+]].d
; VBITS_GE_512: ld1d { [[MASK:z[0-9]+]].d }, [[PG]]/z, [x9]
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: and [[AND:z[0-9]+]].d, [[MASK]].d, #0x1
; VBITS_GE_512-NEXT: cmpne [[COND:p[0-9]+]].d, [[PG1]]/z, [[AND]].d, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].d, [[COND]], [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512: ret
  %mask = load <8 x i1>, <8 x i1>* %c
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %sel = select <8 x i1> %mask, <8 x i64> %op1, <8 x i64> %op2
  store <8 x i64> %sel, <8 x i64>* %a
  ret void
}

define void @select_v16i64(<16 x i64>* %a, <16 x i64>* %b, <16 x i1>* %c) #0 {
; CHECK: select_v16i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; CHECK: ptrue [[PG1:p[0-9]+]].d
; VBITS_GE_1024: ld1d { [[MASK:z[0-9]+]].d }, [[PG]]/z, [x9]
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: and [[AND:z[0-9]+]].d, [[MASK]].d, #0x1
; VBITS_GE_1024-NEXT: cmpne [[COND:p[0-9]+]].d, [[PG1]]/z, [[AND]].d, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].d, [[COND]], [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024: ret
  %mask = load <16 x i1>, <16 x i1>* %c
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %sel = select <16 x i1> %mask, <16 x i64> %op1, <16 x i64> %op2
  store <16 x i64> %sel, <16 x i64>* %a
  ret void
}

define void @select_v32i64(<32 x i64>* %a, <32 x i64>* %b, <32 x i1>* %c) #0 {
; CHECK: select_v32i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; CHECK: ptrue [[PG1:p[0-9]+]].d
; VBITS_GE_2048: ld1d { [[MASK:z[0-9]+]].d }, [[PG]]/z, [x9]
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: and [[AND:z[0-9]+]].d, [[MASK]].d, #0x1
; VBITS_GE_2048-NEXT: cmpne [[COND:p[0-9]+]].d, [[PG1]]/z, [[AND]].d, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].d, [[COND]], [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048: ret
  %mask = load <32 x i1>, <32 x i1>* %c
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %sel = select <32 x i1> %mask, <32 x i64> %op1, <32 x i64> %op2
  store <32 x i64> %sel, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
