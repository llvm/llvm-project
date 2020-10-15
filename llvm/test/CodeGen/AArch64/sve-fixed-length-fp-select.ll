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
define <4 x half> @select_v4f16(<4 x half> %op1, <4 x half> %op2, <4 x i1> %mask) #0 {
; CHECK-LABEL: select_v4f16:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <4 x i1> %mask, <4 x half> %op1, <4 x half> %op2
  ret <4 x half> %sel
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @select_v8f16(<8 x half> %op1, <8 x half> %op2, <8 x i1> %mask) #0 {
; CHECK-LABEL: select_v8f16:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <8 x i1> %mask, <8 x half> %op1, <8 x half> %op2
  ret <8 x half> %sel
}

define void @select_v16f16(<16 x half>* %a, <16 x half>* %b, <16 x i1>* %c) #0 {
; CHECK-LABEL: select_v16f16:
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
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %sel = select <16 x i1> %mask, <16 x half> %op1, <16 x half> %op2
  store <16 x half> %sel, <16 x half>* %a
  ret void
}

define void @select_v32f16(<32 x half>* %a, <32 x half>* %b, <32 x i1>* %c) #0 {
; CHECK-LABEL: select_v32f16:
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
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %sel = select <32 x i1> %mask, <32 x half> %op1, <32 x half> %op2
  store <32 x half> %sel, <32 x half>* %a
  ret void
}

define void @select_v64f16(<64 x half>* %a, <64 x half>* %b, <64 x i1>* %c) #0 {
; CHECK-LABEL: select_v64f16:
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
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %sel = select <64 x i1> %mask, <64 x half> %op1, <64 x half> %op2
  store <64 x half> %sel, <64 x half>* %a
  ret void
}

define void @select_v128f16(<128 x half>* %a, <128 x half>* %b, <128 x i1>* %c) #0 {
; CHECK-LABEL: select_v128f16:
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
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %sel = select <128 x i1> %mask, <128 x half> %op1, <128 x half> %op2
  store <128 x half> %sel, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @select_v2f32(<2 x float> %op1, <2 x float> %op2, <2 x i1> %mask) #0 {
; CHECK-LABEL: select_v2f32:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <2 x i1> %mask, <2 x float> %op1, <2 x float> %op2
  ret <2 x float> %sel
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @select_v4f32(<4 x float> %op1, <4 x float> %op2, <4 x i1> %mask) #0 {
; CHECK-LABEL: select_v4f32:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <4 x i1> %mask, <4 x float> %op1, <4 x float> %op2
  ret <4 x float> %sel
}

define void @select_v8f32(<8 x float>* %a, <8 x float>* %b, <8 x i1>* %c) #0 {
; CHECK-LABEL: select_v8f32:
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
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %sel = select <8 x i1> %mask, <8 x float> %op1, <8 x float> %op2
  store <8 x float> %sel, <8 x float>* %a
  ret void
}

define void @select_v16f32(<16 x float>* %a, <16 x float>* %b, <16 x i1>* %c) #0 {
; CHECK-LABEL: select_v16f32:
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
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %sel = select <16 x i1> %mask, <16 x float> %op1, <16 x float> %op2
  store <16 x float> %sel, <16 x float>* %a
  ret void
}

define void @select_v32f32(<32 x float>* %a, <32 x float>* %b, <32 x i1>* %c) #0 {
; CHECK-LABEL: select_v32f32:
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
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %sel = select <32 x i1> %mask, <32 x float> %op1, <32 x float> %op2
  store <32 x float> %sel, <32 x float>* %a
  ret void
}

define void @select_v64f32(<64 x float>* %a, <64 x float>* %b, <64 x i1>* %c) #0 {
; CHECK-LABEL: select_v64f32:
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
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %sel = select <64 x i1> %mask, <64 x float> %op1, <64 x float> %op2
  store <64 x float> %sel, <64 x float>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @select_v1f64(<1 x double> %op1, <1 x double> %op2, <1 x i1> %mask) #0 {
; CHECK-LABEL: select_v1f64:
; CHECK: bif v0.8b, v1.8b, v2.8b
; CHECK: ret
  %sel = select <1 x i1> %mask, <1 x double> %op1, <1 x double> %op2
  ret <1 x double> %sel
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @select_v2f64(<2 x double> %op1, <2 x double> %op2, <2 x i1> %mask) #0 {
; CHECK-LABEL: select_v2f64:
; CHECK: bif v0.16b, v1.16b, v2.16b
; CHECK: ret
  %sel = select <2 x i1> %mask, <2 x double> %op1, <2 x double> %op2
  ret <2 x double> %sel
}

define void @select_v4f64(<4 x double>* %a, <4 x double>* %b, <4 x i1>* %c) #0 {
; CHECK-LABEL: select_v4f64:
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
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %sel = select <4 x i1> %mask, <4 x double> %op1, <4 x double> %op2
  store <4 x double> %sel, <4 x double>* %a
  ret void
}

define void @select_v8f64(<8 x double>* %a, <8 x double>* %b, <8 x i1>* %c) #0 {
; CHECK-LABEL: select_v8f64:
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
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %sel = select <8 x i1> %mask, <8 x double> %op1, <8 x double> %op2
  store <8 x double> %sel, <8 x double>* %a
  ret void
}

define void @select_v16f64(<16 x double>* %a, <16 x double>* %b, <16 x i1>* %c) #0 {
; CHECK-LABEL: select_v16f64:
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
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %sel = select <16 x i1> %mask, <16 x double> %op1, <16 x double> %op2
  store <16 x double> %sel, <16 x double>* %a
  ret void
}

define void @select_v32f64(<32 x double>* %a, <32 x double>* %b, <32 x i1>* %c) #0 {
; CHECK-LABEL: select_v32f64:
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
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %sel = select <32 x i1> %mask, <32 x double> %op1, <32 x double> %op2
  store <32 x double> %sel, <32 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
