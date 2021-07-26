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
; CHECK-NEXT: st1b { z[[Q0]].d }, p[[P0]], [x1]
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
; CHECK-NEXT: st1b { z[[Q0]].d }, p[[P0]], [x1]
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
; VBITS_GE_512-NEXT: st1b { [[Z0]].d }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_LO:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_HI:z[0-9]+]].d }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s, vl4
; VBITS_EQ_256-DAG: uzp1 [[WORDS_LO:z[0-9]+]].s, [[DWORDS_LO]].s, [[DWORDS_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[WORDS_HI:z[0-9]+]].s, [[DWORDS_HI]].s, [[DWORDS_HI]].s
; VBITS_EQ_256-DAG: splice [[WORDS:z[0-9]+]].s, [[PG2]], [[WORDS_LO]].s, [[WORDS_HI]].s
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1b { [[WORDS]].s }, [[PG3]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i8>
  store <8 x i8> %val, <8 x i8>* %dest
  ret void
}

define void @store_trunc_v16i64i8(<16 x i64>* %ap, <16 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i64i8:
; VBITS_GE_1024: ptrue p[[P0:[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_1024-NEXT: st1b { [[Z0]].d }, p[[P0]], [x1]
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
; VBITS_GE_2048-NEXT: st1b { [[Z0]].d }, p[[P0]], [x1]
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
; VBITS_GE_512-NEXT: st1h { [[Z0]].d }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; Currently does not use the truncating store
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_HI:z[0-9]+]].d }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: uzp1 [[WORDS_LO:z[0-9]+]].s, [[DWORDS_LO]].s, [[DWORDS_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[WORDS_HI:z[0-9]+]].s, [[DWORDS_HI]].s, [[DWORDS_HI]].s
; VBITS_EQ_256-DAG: uzp1 z[[HALFS_LO:[0-9]+]].h, [[WORDS_LO]].h, [[WORDS_LO]].h
; VBITS_EQ_256-DAG: uzp1 z[[HALFS_HI:[0-9]+]].h, [[WORDS_HI]].h, [[WORDS_HI]].h
; VBITS_EQ_256-NEXT: mov v[[HALFS_LO]].d[1], v[[HALFS_HI]].d[0]
; VBITS_EQ_256-NEXT: str q[[HALFS_LO]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i16>
  store <8 x i16> %val, <8 x i16>* %dest
  ret void
}

define void @store_trunc_v8i64i32(<8 x i64>* %ap, <8 x i32>* %dest) #0 {
; CHECK-LABEL: store_trunc_v8i64i32:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1w { [[Z0]].d }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #4
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_LO:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[DWORDS_HI:z[0-9]+]].d }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #3]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].s, vl4
; VBITS_EQ_256-DAG: uzp1 [[WORDS_LO:z[0-9]+]].s, [[DWORDS_LO]].s, [[DWORDS_LO]].s
; VBITS_EQ_256-DAG: uzp1 [[WORDS_HI:z[0-9]+]].s, [[DWORDS_HI]].s, [[DWORDS_HI]].s
; VBITS_EQ_256-DAG: splice [[WORDS:z[0-9]+]].s, [[PG1]], [[WORDS_LO]].s, [[WORDS_HI]].s
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: st1w { [[WORDS]].s }, [[PG3]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %val = trunc <8 x i64> %a to <8 x i32>
  store <8 x i32> %val, <8 x i32>* %dest
  ret void
}

define void @store_trunc_v16i32i8(<16 x i32>* %ap, <16 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i32i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1b { [[Z0]].s }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; Currently does not use the truncating store
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[WORDS_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[WORDS_HI:z[0-9]+]].s }, [[PG]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: uzp1 [[HALFS_LO:z[0-9]+]].h, [[WORDS_LO]].h, [[WORDS_LO]].h
; VBITS_EQ_256-DAG: uzp1 [[HALFS_HI:z[0-9]+]].h, [[WORDS_HI]].h, [[WORDS_HI]].h
; VBITS_EQ_256-DAG: uzp1 z[[BYTES_LO:[0-9]+]].b, [[HALFS_LO]].b, [[HALFS_LO]].b
; VBITS_EQ_256-DAG: uzp1 z[[BYTES_HI:[0-9]+]].b, [[HALFS_HI]].b, [[HALFS_HI]].b
; VBITS_EQ_256-NEXT: mov v[[BYTES_LO]].d[1], v[[BYTES_HI]].d[0]
; VBITS_EQ_256-NEXT: str q[[BYTES_LO]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %val = trunc <16 x i32> %a to <16 x i8>
  store <16 x i8> %val, <16 x i8>* %dest
  ret void
}

define void @store_trunc_v16i32i16(<16 x i32>* %ap, <16 x i16>* %dest) #0 {
; CHECK-LABEL: store_trunc_v16i32i16:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1h { [[Z0]].s }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: ld1w { [[WORDS_LO:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[WORDS_HI:z[0-9]+]].s }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].h, vl8
; VBITS_EQ_256-DAG: uzp1 [[HALFS_LO:z[0-9]+]].h, [[WORDS_LO]].h, [[WORDS_LO]].h
; VBITS_EQ_256-DAG: uzp1 [[HALFS_HI:z[0-9]+]].h, [[WORDS_HI]].h, [[WORDS_HI]].h
; VBITS_EQ_256-DAG: splice [[HALFS:z[0-9]+]].h, [[PG2]], [[HALFS_LO]].h, [[HALFS_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: st1h { [[HALFS]].h }, [[PG3]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %val = trunc <16 x i32> %a to <16 x i16>
  store <16 x i16> %val, <16 x i16>* %dest
  ret void
}

define void @store_trunc_v32i16i8(<32 x i16>* %ap, <32 x i8>* %dest) #0 {
; CHECK-LABEL: store_trunc_v32i16i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: st1b { [[Z0]].h }, p[[P0]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #16
; VBITS_EQ_256-DAG: ld1h { [[HALFS_LO:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[HALFS_HI:z[0-9]+]].h }, [[PG1]]/z, [x0, x[[NUMELTS]], lsl #1]
; VBITS_EQ_256-DAG: ptrue [[PG2:p[0-9]+]].b, vl16
; VBITS_EQ_256-DAG: uzp1 [[BYTES_LO:z[0-9]+]].b, [[HALFS_LO]].b, [[HALFS_LO]].b
; VBITS_EQ_256-DAG: uzp1 [[BYTES_HI:z[0-9]+]].b, [[HALFS_HI]].b, [[HALFS_HI]].b
; VBITS_EQ_256-DAG: splice [[BYTES:z[0-9]+]].b, [[PG2]], [[BYTES_LO]].b, [[BYTES_HI]].b
; VBITS_EQ_256-DAG: ptrue [[PG3:p[0-9]+]].b, vl32
; VBITS_EQ_256-NEXT: st1b { [[BYTES]].b }, [[PG3]], [x1]
; VBITS_EQ_256-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %val = trunc <32 x i16> %a to <32 x i8>
  store <32 x i8> %val, <32 x i8>* %dest
  ret void
}

attributes #0 = { "target-features"="+sve" }
