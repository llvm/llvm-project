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
define void @store_v2f32(<2 x float>* %a) #0 {
; CHECK-LABEL: store_v2f32:
; CHECK: str xzr, [x0]
; CHECK: ret
  store <2 x float> zeroinitializer, <2 x float>* %a
  ret void
}

; Don't use SVE for 128-bit vectors.
define void @store_v4f32(<4 x float>* %a) #0 {
; CHECK-LABEL: store_v4f32:
; CHECK: stp xzr, xzr, [x0]
; CHECK: ret
  store <4 x float> zeroinitializer, <4 x float>* %a
  ret void
}

define void @store_v8f32(<8 x float>* %a) #0 {
; CHECK-LABEL: store_v8f32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),8)]]
; CHECK: st1w { z{{[0-9]+}}.s }, [[PG]], [x0]
; CHECK: ret
  store <8 x float> zeroinitializer, <8 x float>* %a
  ret void
}

define void @store_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: store_v16f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),16)]]
; CHECK-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0]
; VBITS_LE_256-DAG: mov x[[A1:[0-9]+]], #[[#div(VBYTES,4)]]
; VBITS_LE_256-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A1]], lsl #2]
; CHECK: ret
  store <16 x float> zeroinitializer, <16 x float>* %a
  ret void
}

define void @store_v32f32(<32 x float>* %a) #0 {
; CHECK-LABEL: store_v32f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),32)]]
; CHECK-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0]
; VBITS_LE_512-DAG: mov x[[A1:[0-9]+]], #[[#div(VBYTES,4)]]
; VBITS_LE_512-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A1]], lsl #2]
; VBITS_LE_256-DAG: mov x[[A2:[0-9]+]], #[[#mul(div(VBYTES,4),2)]]
; VBITS_LE_256-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A2]], lsl #2]
; VBITS_LE_256-DAG: mov x[[A3:[0-9]+]], #[[#mul(div(VBYTES,4),3)]]
; VBITS_LE_256-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A3]], lsl #2]
; CHECK: ret
  store <32 x float> zeroinitializer, <32 x float>* %a
  ret void
}

define void @store_v64f32(<64 x float>* %a) #0 {
; CHECK-LABEL: store_v64f32:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl[[#min(div(VBYTES,4),64)]]
; CHECK-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0]
; VBITS_LE_1024-DAG: mov x[[A1:[0-9]+]], #[[#div(VBYTES,4)]]
; VBITS_LE_1024-DAG: st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A1]], lsl #2]
; VBITS_LE_512-DAG:  mov x[[A2:[0-9]+]], #[[#mul(div(VBYTES,4),2)]]
; VBITS_LE_512-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A2]], lsl #2]
; VBITS_LE_512-DAG:  mov x[[A3:[0-9]+]], #[[#mul(div(VBYTES,4),3)]]
; VBITS_LE_512-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A3]], lsl #2]
; VBITS_LE_256-DAG:  mov x[[A4:[0-9]+]], #[[#mul(div(VBYTES,4),4)]]
; VBITS_LE_256-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A4]], lsl #2]
; VBITS_LE_256-DAG:  mov x[[A5:[0-9]+]], #[[#mul(div(VBYTES,4),5)]]
; VBITS_LE_256-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A5]], lsl #2]
; VBITS_LE_256-DAG:  mov x[[A6:[0-9]+]], #[[#mul(div(VBYTES,4),6)]]
; VBITS_LE_256-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A6]], lsl #2]
; VBITS_LE_256-DAG:  mov x[[A7:[0-9]+]], #[[#mul(div(VBYTES,4),7)]]
; VBITS_LE_256-DAG:  st1w { z{{[0-9]+}}.s }, [[PG]], [x0, x[[A7]], lsl #2]
; CHECK: ret
  store <64 x float> zeroinitializer, <64 x float>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
