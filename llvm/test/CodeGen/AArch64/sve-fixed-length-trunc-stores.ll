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

define void @store_trunc_v2i64i8(<2 x i64>* %ap, <2 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v2i64i8
; CHECK: ldr q[[Q0:[0-9]+]], [x0]
; CHECK: ptrue p[[P0:[0-9]+]].d, vl2
; CHECK-NEXT: st1b { z[[Q0]].d }, p[[P0]], [x{{[0-9]+}}]
; CHECK-NEXT: ret
  %a = load <2 x i64>, <2 x i64>* %ap
  %val = trunc <2 x i64> %a to <2 x i8>
  store <2 x i8> %val, <2 x i8>* %dest
  ret void
}

define void @store_trunc_v4i64i8(<4 x i64>* %ap, <4 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v4i64i8
; CHECK: ptrue p[[P0:[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; CHECK-NEXT: st1b { z[[Q0]].d }, p[[P0]], [x{{[0-9]+}}]
; CHECK-NEXT: ret
  %a = load <4 x i64>, <4 x i64>* %ap
  %val = trunc <4 x i64> %a to <4 x i8>
  store <4 x i8> %val, <4 x i8>* %dest
  ret void
}

define void @store_trunc_v8i64i8(<8 x i64>* %ap, <8 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v8i64i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1b { [[Z0]].d }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: ld1d { [[Z0:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1d { [[Z1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG]].s, vl4
; VBITS_EQ_256-DAG: uzp1 [[Z0]].s, [[Z0]].s, [[Z0]].s
; VBITS_EQ_256-DAG: uzp1 [[Z1]].s, [[Z1]].s, [[Z1]].s
; VBITS_EQ_256-DAG: splice [[Z1]].s, [[PG]], [[Z1]].s, [[Z0]].s
; VBITS_EQ_256-DAG: ptrue [[PG]].s, vl8
; VBITS_EQ_256-DAG: st1b { [[Z1]].s }, [[PG]], [x1]
; VBITS_EQ_256-DAG: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i8>
  store <8 x i8> %val, <8 x i8>* %dest
  ret void
}

define void @store_trunc_v16i64i8(<16 x i64>* %ap, <16 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i64i8:
; VBITS_GE_1024: ptrue p[[P0:[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_1024-NEXT: st1b { [[Z0]].d }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i64>, <16 x i64>* %ap
  %val = trunc <16 x i64> %a to <16 x i8>
  store <16 x i8> %val, <16 x i8>* %dest
  ret void
}

define void @store_trunc_v32i64i8(<32 x i64>* %ap, <32 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v32i64i8:
; VBITS_GE_2048: ptrue p[[P0:[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_2048-NEXT: st1b { [[Z0]].d }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i64>, <32 x i64>* %ap
  %val = trunc <32 x i64> %a to <32 x i8>
  store <32 x i8> %val, <32 x i8>* %dest
  ret void
}

define void @store_trunc_v8i64i16(<8 x i64>* %ap, <8 x i16>* %dest) #0 {
; CHECK-LABEL: store_trunc_v8i64i16:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1h { [[Z0]].d }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; Currently does not use the truncating store
; VBITS_EQ_256-DAG:	ptrue	[[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: ld1d	{ [[Z0:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1d	{ [[Z1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: uzp1	[[Z0]].s, [[Z0]].s, [[Z0]].s
; VBITS_EQ_256-DAG: uzp1	[[Z1]].s, [[Z1]].s, [[Z1]].s
; VBITS_EQ_256-DAG: uzp1	[[Z1]].h, [[Z1]].h, [[Z1]].h
; VBITS_EQ_256-DAG:	uzp1	[[Z0]].h, [[Z0]].h, [[Z0]].h
; VBITS_EQ_256-DAG:	mov	v[[V0:[0-9]+]].d[1], v{{[0-9]+}}.d[0]
; VBITS_EQ_256-DAG:	str	q[[V0]], [x1]
; VBITS_EQ_256-DAG:	ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i16>
  store <8 x i16> %val, <8 x i16>* %dest
  ret void
}

define void @store_trunc_v8i64i32(<8 x i64>* %ap, <8 x i32>* %dest) #0 {
; CHECK-LABEL: store_trunc_v8i64i32:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1w { [[Z0]].d }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: ld1d { [[Z0:z[0-9]+]].d }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1d { [[Z1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG]].s, vl4
; VBITS_EQ_256-DAG: uzp1 [[Z0]].s, [[Z0]].s, [[Z0]].s
; VBITS_EQ_256-DAG: uzp1 [[Z1]].s, [[Z1]].s, [[Z1]].s
; VBITS_EQ_256-DAG: splice [[Z1]].s, [[PG]], [[Z1]].s, [[Z0]].s
; VBITS_EQ_256-DAG: ptrue [[PG]].s, vl8
; VBITS_EQ_256-DAG: st1w { [[Z1]].s }, [[PG]], [x1]
; VBITS_EQ_256-DAG: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i32>
  store <8 x i32> %val, <8 x i32>* %dest
  ret void
}

define void @store_trunc_v16i32i8(<16 x i32>* %ap, <16 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i32i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1b { [[Z0]].s }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; Currently does not use the truncating store
; VBITS_EQ_256-DAG:	ptrue	[[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w	{ [[Z0:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1w	{ [[Z1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: uzp1	[[Z0]].h, [[Z0]].h, [[Z0]].h
; VBITS_EQ_256-DAG: uzp1	[[Z1]].h, [[Z1]].h, [[Z1]].h
; VBITS_EQ_256-DAG: uzp1	[[Z1]].b, [[Z1]].b, [[Z1]].b
; VBITS_EQ_256-DAG:	uzp1	[[Z0]].b, [[Z0]].b, [[Z0]].b
; VBITS_EQ_256-DAG:	mov	v[[V0:[0-9]+]].d[1], v{{[0-9]+}}.d[0]
; VBITS_EQ_256-DAG:	str	q[[V0]], [x1]
; VBITS_EQ_256-DAG:	ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %val = trunc <16 x i32> %a to <16 x i8>
  store <16 x i8> %val, <16 x i8>* %dest
  ret void
}

define void @store_trunc_v16i32i16(<16 x i32>* %ap, <16 x i16>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i32i16:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1h { [[Z0]].s }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: ld1w { [[Z0:z[0-9]+]].s }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1w { [[Z1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG]].h, vl8
; VBITS_EQ_256-DAG: uzp1 [[Z0]].h, [[Z0]].h, [[Z0]].h
; VBITS_EQ_256-DAG: uzp1 [[Z1]].h, [[Z1]].h, [[Z1]].h
; VBITS_EQ_256-DAG: splice [[Z1]].h, [[PG]], [[Z1]].h, [[Z0]].h
; VBITS_EQ_256-DAG: ptrue [[PG]].h, vl16
; VBITS_EQ_256-DAG: st1h { [[Z1]].h }, [[PG]], [x1]
; VBITS_EQ_256-DAG: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %val = trunc <16 x i32> %a to <16 x i16>
  store <16 x i16> %val, <16 x i16>* %dest
  ret void
}

define void @store_trunc_v32i16i8(<32 x i16>* %ap, <32 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v32i16i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1b { [[Z0]].h }, p[[P0]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: ld1h { [[Z0:z[0-9]+]].h }, [[PG]]/z, [x8]
; VBITS_EQ_256-DAG: ld1h { [[Z1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ptrue [[PG]].b, vl16
; VBITS_EQ_256-DAG: uzp1 [[Z0]].b, [[Z0]].b, [[Z0]].b
; VBITS_EQ_256-DAG: uzp1 [[Z1]].b, [[Z1]].b, [[Z1]].b
; VBITS_EQ_256-DAG: splice [[Z1]].b, [[PG]], [[Z1]].b, [[Z0]].b
; VBITS_EQ_256-DAG: ptrue [[PG]].b, vl32
; VBITS_EQ_256-DAG: st1b { [[Z1]].b }, [[PG]], [x1]
; VBITS_EQ_256-DAG: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = trunc <32 x i16> %a to <32 x i8>
  store <32 x i8> %val, <32 x i8>* %dest
  ret void
}


attributes #0 = { "target-features"="+sve" }
