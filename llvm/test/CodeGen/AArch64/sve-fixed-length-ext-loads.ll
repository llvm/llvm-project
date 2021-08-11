; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_2048,VBITS_GE_1024,VBITS_GE_512

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

define <4 x i32> @load_zext_v4i16i32(<4 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v4i16i32
  ; CHECK: ldr d[[D0:[0-9]+]], [x0]
  ; CHECK-NEXT: ushll v[[D0]].4s, v[[D0]].4h, #0
  ; CHECK-NEXT: ret
  %a = load <4 x i16>, <4 x i16>* %ap
  %val = zext <4 x i16> %a to <4 x i32>
  ret <4 x i32> %val
}

define <8 x i32> @load_zext_v8i16i32(<8 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v8i16i32
  ; CHECK: ptrue [[P0:p[0-9]+]].s, vl8
  ; CHECK-NEXT: ld1h { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; CHECK-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; CHECK-NEXT: ret
  %a = load <8 x i16>, <8 x i16>* %ap
  %val = zext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %val
}

define <16 x i32> @load_zext_v16i16i32(<16 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v16i16i32
  ; VBITS_GE_512: ptrue [[P0:p[0-9]+]].s, vl16
  ; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_512-NEXT: ret

  ; Ensure sensible type legalistaion
  ; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
  ; VBITS_EQ_256-DAG: ld1h { [[Z0:z[0-9]+]].h }, [[PG]]/z, [x0]
  ; VBITS_EQ_256-DAG: mov x9, sp
  ; VBITS_EQ_256-DAG: st1h { [[Z0]].h }, [[PG]], [x9]
  ; VBITS_EQ_256-DAG: ldp q[[R0:[0-9]+]], q[[R1:[0-9]+]], [sp]
  ; VBITS_EQ_256-DAG: add x9, x8, #32
  ; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
  ; VBITS_EQ_256-DAG: uunpklo z[[R0]].s, z[[R0]].h
  ; VBITS_EQ_256-DAG: uunpklo z[[R1]].s, z[[R1]].h
  ; VBITS_EQ_256-DAG: st1w { z[[R1]].s }, [[PG1]], [x9]
  ; VBITS_EQ_256-DAG: st1w { z[[R0]].s }, [[PG1]], [x8]
  ; VBITS_EQ_256-DAG: ret
  %a = load <16 x i16>, <16 x i16>* %ap
  %val = zext <16 x i16> %a to <16 x i32>
  ret <16 x i32> %val
}

define <32 x i32> @load_zext_v32i16i32(<32 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v32i16i32
  ; VBITS_GE_1024: ptrue [[P0:p[0-9]+]].s, vl32
  ; VBITS_GE_1024-NEXT: ld1h { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_1024-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = zext <32 x i16> %a to <32 x i32>
  ret <32 x i32> %val
}

define <64 x i32> @load_zext_v64i16i32(<64 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v64i16i32
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].s, vl64
  ; VBITS_GE_2048-NEXT: ld1h { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i16>, <64 x i16>* %ap
  %val = zext <64 x i16> %a to <64 x i32>
  ret <64 x i32> %val
}

define <4 x i32> @load_sext_v4i16i32(<4 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v4i16i32
  ; CHECK: ldr d[[D0:[0-9]+]], [x0]
  ; CHECK-NEXT: sshll v[[D0]].4s, v[[D0]].4h, #0
  ; CHECK-NEXT: ret
  %a = load <4 x i16>, <4 x i16>* %ap
  %val = sext <4 x i16> %a to <4 x i32>
  ret <4 x i32> %val
}

define <8 x i32> @load_sext_v8i16i32(<8 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v8i16i32
  ; CHECK: ptrue [[P0:p[0-9]+]].s, vl8
  ; CHECK-NEXT: ld1sh { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; CHECK-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; CHECK-NEXT: ret
  %a = load <8 x i16>, <8 x i16>* %ap
  %val = sext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %val
}

define <16 x i32> @load_sext_v16i16i32(<16 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v16i16i32
  ; VBITS_GE_512: ptrue [[P0:p[0-9]+]].s, vl16
  ; VBITS_GE_512-NEXT: ld1sh { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_512-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_512-NEXT: ret

  ; Ensure sensible type legalistaion
  ; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
  ; VBITS_EQ_256-DAG: ld1h { [[Z0:z[0-9]+]].h }, [[PG]]/z, [x0]
  ; VBITS_EQ_256-DAG: mov x9, sp
  ; VBITS_EQ_256-DAG: st1h { [[Z0]].h }, [[PG]], [x9]
  ; VBITS_EQ_256-DAG: ldp q[[R0:[0-9]+]], q[[R1:[0-9]+]], [sp]
  ; VBITS_EQ_256-DAG: add x9, x8, #32
  ; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
  ; VBITS_EQ_256-DAG: sunpklo z[[R0]].s, z[[R0]].h
  ; VBITS_EQ_256-DAG: sunpklo z[[R1]].s, z[[R1]].h
  ; VBITS_EQ_256-DAG: st1w { z[[R1]].s }, [[PG1]], [x9]
  ; VBITS_EQ_256-DAG: st1w { z[[R0]].s }, [[PG1]], [x8]
  ; VBITS_EQ_256-DAG: ret
  %a = load <16 x i16>, <16 x i16>* %ap
  %val = sext <16 x i16> %a to <16 x i32>
  ret <16 x i32> %val
}

define <32 x i32> @load_sext_v32i16i32(<32 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v32i16i32
  ; VBITS_GE_1024: ptrue [[P0:p[0-9]+]].s, vl32
  ; VBITS_GE_1024-NEXT: ld1sh { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_1024-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = sext <32 x i16> %a to <32 x i32>
  ret <32 x i32> %val
}

define <64 x i32> @load_sext_v64i16i32(<64 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v64i16i32
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].s, vl64
  ; VBITS_GE_2048-NEXT: ld1sh { [[Z0:z[0-9]+]].s }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1w { [[Z0]].s }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i16>, <64 x i16>* %ap
  %val = sext <64 x i16> %a to <64 x i32>
  ret <64 x i32> %val
}

define <32 x i64> @load_zext_v32i8i64(<32 x i8>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v32i8i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1b { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %ap
  %val = zext <32 x i8> %a to <32 x i64>
  ret <32 x i64> %val
}

define <32 x i64> @load_sext_v32i8i64(<32 x i8>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v32i8i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1sb { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %ap
  %val = sext <32 x i8> %a to <32 x i64>
  ret <32 x i64> %val
}

define <32 x i64> @load_zext_v32i16i64(<32 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v32i16i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1h { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = zext <32 x i16> %a to <32 x i64>
  ret <32 x i64> %val
}

define <32 x i64> @load_sext_v32i16i64(<32 x i16>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v32i16i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1sh { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = sext <32 x i16> %a to <32 x i64>
  ret <32 x i64> %val
}

define <32 x i64> @load_zext_v32i32i64(<32 x i32>* %ap) #0 {
  ; CHECK-LABEL: load_zext_v32i32i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1w { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i32>, <32 x i32>* %ap
  %val = zext <32 x i32> %a to <32 x i64>
  ret <32 x i64> %val
}

define <32 x i64> @load_sext_v32i32i64(<32 x i32>* %ap) #0 {
  ; CHECK-LABEL: load_sext_v32i32i64
  ; VBITS_GE_2048: ptrue [[P0:p[0-9]+]].d, vl32
  ; VBITS_GE_2048-NEXT: ld1sw { [[Z0:z[0-9]+]].d }, [[P0]]/z, [x0]
  ; VBITS_GE_2048-NEXT: st1d { [[Z0]].d }, [[P0]], [x8]
  ; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i32>, <32 x i32>* %ap
  %val = sext <32 x i32> %a to <32 x i64>
  ret <32 x i64> %val
}

attributes #0 = { "target-features"="+sve" }
