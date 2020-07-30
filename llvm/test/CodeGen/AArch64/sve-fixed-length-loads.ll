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
define <2 x float> @load_v2f32(<2 x float>* %a) #0 {
; CHECK-LABEL: load_v2f32:
; CHECK: ldr d0, [x0]
; CHECK: ret
  %load = load <2 x float>, <2 x float>* %a
  ret <2 x float> %load
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @load_v4f32(<4 x float>* %a) #0 {
; CHECK-LABEL: load_v4f32:
; CHECK: ldr q0, [x0]
; CHECK: ret
  %load = load <4 x float>, <4 x float>* %a
  ret <4 x float> %load
}

define <8 x float> @load_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: load_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x0]
; CHECK: ret
  %load = load <8 x float>, <8 x float>* %a
  ret <8 x float> %load
}

define <16 x float> @load_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: load_v16f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x0]
; VBITS_LE_256-DAG: add x[[A1:[0-9]+]], x0, #[[#VBYTES]]
; VBITS_LE_256-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A1]]]
; CHECK: ret
  %load = load <16 x float>, <16 x float>* %a
  ret <16 x float> %load
}

define <32 x float> @load_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: load_v32f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x0]
; VBITS_LE_512-DAG: add x[[A1:[0-9]+]], x0, #[[#VBYTES]]
; VBITS_LE_512-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A1]]]
; VBITS_LE_256-DAG: add x[[A2:[0-9]+]], x0, #[[#mul(VBYTES,2)]]
; VBITS_LE_256-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A2]]]
; VBITS_LE_256-DAG: add x[[A3:[0-9]+]], x0, #[[#mul(VBYTES,3)]]
; VBITS_LE_256-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A3]]]
; CHECK: ret
  %load = load <32 x float>, <32 x float>* %a
  ret <32 x float> %load
}

define <64 x float> @load_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: load_v64f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x0]
; VBITS_LE_1024-DAG: add x[[A1:[0-9]+]], x0, #[[#VBYTES]]
; VBITS_LE_1024-DAG: ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A1]]]
; VBITS_LE_512-DAG:  add x[[A2:[0-9]+]], x0, #[[#mul(VBYTES,2)]]
; VBITS_LE_512-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A2]]]
; VBITS_LE_512-DAG:  add x[[A3:[0-9]+]], x0, #[[#mul(VBYTES,3)]]
; VBITS_LE_512-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A3]]]
; VBITS_LE_256-DAG:  add x[[A4:[0-9]+]], x0, #[[#mul(VBYTES,4)]]
; VBITS_LE_256-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A4]]]
; VBITS_LE_256-DAG:  add x[[A5:[0-9]+]], x0, #[[#mul(VBYTES,5)]]
; VBITS_LE_256-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A5]]]
; VBITS_LE_256-DAG:  add x[[A6:[0-9]+]], x0, #[[#mul(VBYTES,6)]]
; VBITS_LE_256-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A6]]]
; VBITS_LE_256-DAG:  add x[[A7:[0-9]+]], x0, #[[#mul(VBYTES,7)]]
; VBITS_LE_256-DAG:  ld1w { z{{[0-9]+}}.s }, [[PG]]/z, [x[[A7]]]
; CHECK: ret
  %load = load <64 x float>, <64 x float>* %a
  ret <64 x float> %load
}

attributes #0 = { "target-features"="+sve" }
