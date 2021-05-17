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
define <4 x half> @select_v4f16(<4 x half> %op1, <4 x half> %op2, i1 %mask) #0 {
; CHECK: select_v4f16:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.4h, w8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <4 x half> %op1, <4 x half> %op2
  ret <4 x half> %sel
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @select_v8f16(<8 x half> %op1, <8 x half> %op2, i1 %mask) #0 {
; CHECK: select_v8f16:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.8h, w8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <8 x half> %op1, <8 x half> %op2
  ret <8 x half> %sel
}

define void @select_v16f16(<16 x half>* %a, <16 x half>* %b, i1 %mask) #0 {
; CHECK: select_v16f16:
; CHECK: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),16)]]
; CHECK-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].h
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; CHECK-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <16 x half>, <16 x half>* %a
  %op2 = load volatile <16 x half>, <16 x half>* %b
  %sel = select i1 %mask, <16 x half> %op1, <16 x half> %op2
  store <16 x half> %sel, <16 x half>* %a
  ret void
}

define void @select_v32f16(<32 x half>* %a, <32 x half>* %b, i1 %mask) #0 {
; CHECK: select_v32f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),32)]]
; VBITS_GE_512-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <32 x half>, <32 x half>* %a
  %op2 = load volatile <32 x half>, <32 x half>* %b
  %sel = select i1 %mask, <32 x half> %op1, <32 x half> %op2
  store <32 x half> %sel, <32 x half>* %a
  ret void
}

define void @select_v64f16(<64 x half>* %a, <64 x half>* %b, i1 %mask) #0 {
; CHECK: select_v64f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),64)]]
; VBITS_GE_1024-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <64 x half>, <64 x half>* %a
  %op2 = load volatile <64 x half>, <64 x half>* %b
  %sel = select i1 %mask, <64 x half> %op1, <64 x half> %op2
  store <64 x half> %sel, <64 x half>* %a
  ret void
}

define void @select_v128f16(<128 x half>* %a, <128 x half>* %b, i1 %mask) #0 {
; CHECK: select_v128f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl[[#min(div(VBYTES,2),128)]]
; VBITS_GE_2048-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].h, w[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].h, [[TMP1]].h, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].h
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].h, [[PG2]]/z, [[TMP2]].h, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].h, [[PRES]], [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <128 x half>, <128 x half>* %a
  %op2 = load volatile <128 x half>, <128 x half>* %b
  %sel = select i1 %mask, <128 x half> %op1, <128 x half> %op2
  store <128 x half> %sel, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @select_v2f32(<2 x float> %op1, <2 x float> %op2, i1 %mask) #0 {
; CHECK: select_v2f32:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.2s, w8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <2 x float> %op1, <2 x float> %op2
  ret <2 x float> %sel
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @select_v4f32(<4 x float> %op1, <4 x float> %op2, i1 %mask) #0 {
; CHECK: select_v4f32:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm w8, ne
; CHECK-NEXT: dup v2.4s, w8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <4 x float> %op1, <4 x float> %op2
  ret <4 x float> %sel
}

define void @select_v8f32(<8 x float>* %a, <8 x float>* %b, i1 %mask) #0 {
; CHECK: select_v8f32:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <8 x float>, <8 x float>* %a
  %op2 = load volatile <8 x float>, <8 x float>* %b
  %sel = select i1 %mask, <8 x float> %op1, <8 x float> %op2
  store <8 x float> %sel, <8 x float>* %a
  ret void
}

define void @select_v16f32(<16 x float>* %a, <16 x float>* %b, i1 %mask) #0 {
; CHECK: select_v16f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; VBITS_GE_512-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <16 x float>, <16 x float>* %a
  %op2 = load volatile <16 x float>, <16 x float>* %b
  %sel = select i1 %mask, <16 x float> %op1, <16 x float> %op2
  store <16 x float> %sel, <16 x float>* %a
  ret void
}

define void @select_v32f32(<32 x float>* %a, <32 x float>* %b, i1 %mask) #0 {
; CHECK: select_v32f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; VBITS_GE_1024-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <32 x float>, <32 x float>* %a
  %op2 = load volatile <32 x float>, <32 x float>* %b
  %sel = select i1 %mask, <32 x float> %op1, <32 x float> %op2
  store <32 x float> %sel, <32 x float>* %a
  ret void
}

define void @select_v64f32(<64 x float>* %a, <64 x float>* %b, i1 %mask) #0 {
; CHECK: select_v64f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; VBITS_GE_2048-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].s, w[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].s, [[TMP1]].s, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].s, [[PG2]]/z, [[TMP2]].s, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].s, [[PRES]], [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <64 x float>, <64 x float>* %a
  %op2 = load volatile <64 x float>, <64 x float>* %b
  %sel = select i1 %mask, <64 x float> %op1, <64 x float> %op2
  store <64 x float> %sel, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @select_v1f64(<1 x double> %op1, <1 x double> %op2, i1 %mask) #0 {
; CHECK: select_v1f64:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm x8, ne
; CHECK-NEXT: fmov d2, x8
; CHECK-NEXT: bif v0.8b, v1.8b, v2.8b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <1 x double> %op1, <1 x double> %op2
  ret <1 x double> %sel
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @select_v2f64(<2 x double> %op1, <2 x double> %op2, i1 %mask) #0 {
; CHECK: select_v2f64:
; CHECK: tst w0, #0x1
; CHECK-NEXT: csetm x8, ne
; CHECK-NEXT: dup v2.2d, x8
; CHECK-NEXT: bif v0.16b, v1.16b, v2.16b
; CHECK-NEXT: ret
  %sel = select i1 %mask, <2 x double> %op1, <2 x double> %op2
  ret <2 x double> %sel
}

define void @select_v4f64(<4 x double>* %a, <4 x double>* %b, i1 %mask) #0 {
; CHECK: select_v4f64:
; CHECK: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),4)]]
; CHECK-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; CHECK-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; CHECK-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; CHECK-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; CHECK-NEXT: ret
  %op1 = load volatile <4 x double>, <4 x double>* %a
  %op2 = load volatile <4 x double>, <4 x double>* %b
  %sel = select i1 %mask, <4 x double> %op1, <4 x double> %op2
  store <4 x double> %sel, <4 x double>* %a
  ret void
}

define void @select_v8f64(<8 x double>* %a, <8 x double>* %b, i1 %mask) #0 {
; CHECK: select_v8f64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),8)]]
; VBITS_GE_512-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_512-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_512-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_512-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load volatile <8 x double>, <8 x double>* %a
  %op2 = load volatile <8 x double>, <8 x double>* %b
  %sel = select i1 %mask, <8 x double> %op1, <8 x double> %op2
  store <8 x double> %sel, <8 x double>* %a
  ret void
}

define void @select_v16f64(<16 x double>* %a, <16 x double>* %b, i1 %mask) #0 {
; CHECK: select_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),16)]]
; VBITS_GE_1024-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_1024-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_1024-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_1024-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load volatile <16 x double>, <16 x double>* %a
  %op2 = load volatile <16 x double>, <16 x double>* %b
  %sel = select i1 %mask, <16 x double> %op1, <16 x double> %op2
  store <16 x double> %sel, <16 x double>* %a
  ret void
}

define void @select_v32f64(<32 x double>* %a, <32 x double>* %b, i1 %mask) #0 {
; CHECK: select_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl[[#min(div(VBYTES,8),32)]]
; VBITS_GE_2048-NEXT: and w[[AND:[0-9]+]], w2, #0x1
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: mov [[TMP1:z[0-9]+]].d, x[[AND]]
; VBITS_GE_2048-NEXT: and [[TMP2:z[0-9]+]].d, [[TMP1]].d, #0x1
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d
; VBITS_GE_2048-NEXT: cmpne [[PRES:p[0-9]+]].d, [[PG2]]/z, [[TMP2]].d, #0
; VBITS_GE_2048-NEXT: sel [[RES:z[0-9]+]].d, [[PRES]], [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG1]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load volatile <32 x double>, <32 x double>* %a
  %op2 = load volatile <32 x double>, <32 x double>* %b
  %sel = select i1 %mask, <32 x double> %op1, <32 x double> %op2
  store <32 x double> %sel, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
