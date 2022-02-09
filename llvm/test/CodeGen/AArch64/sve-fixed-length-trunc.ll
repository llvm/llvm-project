; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: z{0-9}

;
; truncate i16 -> i8
;

define <16 x i8> @trunc_v16i16_v16i8(<16 x i16>* %in) #0 {
; CHECK-LABEL: trunc_v16i16_v16i8:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: ld1h { [[A_HALFS:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 z0.b, [[A_HALFS]].b, [[A_HALFS]].b
; CHECK-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %in
  %b = trunc <16 x i16> %a to <16 x i8>
  ret <16 x i8> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i16_v32i8(<32 x i16>* %in, <32 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v32i16_v32i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512: ld1h { [[A_HALFS:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_512: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <32 x i16>, <32 x i16>* %in
  %b = trunc <32 x i16> %a to <32 x i8>
  %c = add <32 x i8> %b, %b
  store <32 x i8> %c, <32 x i8>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v64i16_v64i8(<64 x i16>* %in, <64 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v64i16_v64i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024: ld1h { [[A_HALFS:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_1024: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <64 x i16>, <64 x i16>* %in
  %b = trunc <64 x i16> %a to <64 x i8>
  %c = add <64 x i8> %b, %b
  store <64 x i8> %c, <64 x i8>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v128i16_v128i8(<128 x i16>* %in, <128 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v128i16_v128i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048: ld1h { [[A_HALFS:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_2048: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <128 x i16>, <128 x i16>* %in
  %b = trunc <128 x i16> %a to <128 x i8>
  %c = add <128 x i8> %b, %b
  store <128 x i8> %c, <128 x i8>* %out
  ret void
}

;
; truncate i32 -> i8
;

define <8 x i8> @trunc_v8i32_v8i8(<8 x i32>* %in) #0 {
; CHECK-LABEL: trunc_v8i32_v8i8:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; CHECK-NEXT: uzp1 z0.b, [[A_HALFS]].b, [[A_HALFS]].b
; CHECK-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %in
  %b = trunc <8 x i32> %a to <8 x i8>
  ret <8 x i8> %b
}

define <16 x i8> @trunc_v16i32_v16i8(<16 x i32>* %in) #0 {
; CHECK-LABEL: trunc_v16i32_v16i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_512-NEXT: uzp1 z0.b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %in
  %b = trunc <16 x i32> %a to <16 x i8>
  ret <16 x i8> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i32_v32i8(<32 x i32>* %in, <32 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v32i32_v32i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_1024: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_1024: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <32 x i32>, <32 x i32>* %in
  %b = trunc <32 x i32> %a to <32 x i8>
  %c = add <32 x i8> %b, %b
  store <32 x i8> %c, <32 x i8>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v64i32_v64i8(<64 x i32>* %in, <64 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v64i32_v64i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_2048: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_2048: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <64 x i32>, <64 x i32>* %in
  %b = trunc <64 x i32> %a to <64 x i8>
  %c = add <64 x i8> %b, %b
  store <64 x i8> %c, <64 x i8>* %out
  ret void
}

;
; truncate i32 -> i16
;

define <8 x i16> @trunc_v8i32_v8i16(<8 x i32>* %in) #0 {
; CHECK-LABEL: trunc_v8i32_v8i16:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 z0.h, [[A_WORDS]].h, [[A_WORDS]].h
; CHECK-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %in
  %b = trunc <8 x i32> %a to <8 x i16>
  ret <8 x i16> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v16i32_v16i16(<16 x i32>* %in, <16 x i16>* %out) #0 {
; CHECK-LABEL: trunc_v16i32_v16i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_512: add [[A_HALFS]].h, [[A_HALFS]].h, [[A_HALFS]].h
  %a = load <16 x i32>, <16 x i32>* %in
  %b = trunc <16 x i32> %a to <16 x i16>
  %c = add <16 x i16> %b, %b
  store <16 x i16> %c, <16 x i16>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i32_v32i16(<32 x i32>* %in, <32 x i16>* %out) #0 {
; CHECK-LABEL: trunc_v32i32_v32i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_1024: add [[A_HALFS]].h, [[A_HALFS]].h, [[A_HALFS]].h
  %a = load <32 x i32>, <32 x i32>* %in
  %b = trunc <32 x i32> %a to <32 x i16>
  %c = add <32 x i16> %b, %b
  store <32 x i16> %c, <32 x i16>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v64i32_v64i16(<64 x i32>* %in, <64 x i16>* %out) #0 {
; CHECK-LABEL: trunc_v64i32_v64i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048: ld1w { [[A_WORDS:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_2048: add [[A_HALFS]].h, [[A_HALFS]].h, [[A_HALFS]].h
  %a = load <64 x i32>, <64 x i32>* %in
  %b = trunc <64 x i32> %a to <64 x i16>
  %c = add <64 x i16> %b, %b
  store <64 x i16> %c, <64 x i16>* %out
  ret void
}

;
; truncate i64 -> i8
;

; NOTE: v4i8 is not legal so result i8 elements are held within i16 containers.
define <4 x i8> @trunc_v4i64_v4i8(<4 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v4i64_v4i8:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; CHECK-NEXT: uzp1 z0.h, [[A_WORDS]].h, [[A_WORDS]].h
; CHECK-NEXT: ret
  %a = load <4 x i64>, <4 x i64>* %in
  %b = trunc <4 x i64> %a to <4 x i8>
  ret <4 x i8> %b
}

define <8 x i8> @trunc_v8i64_v8i8(<8 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v8i64_v8i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_512-NEXT: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_512-NEXT: uzp1 z0.b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %in
  %b = trunc <8 x i64> %a to <8 x i8>
  ret <8 x i8> %b
}

define <16 x i8> @trunc_v16i64_v16i8(<16 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v16i64_v16i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_1024-NEXT: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_1024-NEXT: uzp1 z0.b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i64>, <16 x i64>* %in
  %b = trunc <16 x i64> %a to <16 x i8>
  ret <16 x i8> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i64_v32i8(<32 x i64>* %in, <32 x i8>* %out) #0 {
; CHECK-LABEL: trunc_v32i64_v32i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_2048: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_2048: uzp1 [[A_BYTES:z[0-9]+]].b, [[A_HALFS]].b, [[A_HALFS]].b
; VBITS_GE_2048: add [[A_BYTES]].b, [[A_BYTES]].b, [[A_BYTES]].b
  %a = load <32 x i64>, <32 x i64>* %in
  %b = trunc <32 x i64> %a to <32 x i8>
  %c = add <32 x i8> %b, %b
  store <32 x i8> %c, <32 x i8>* %out
  ret void
}

;
; truncate i64 -> i16
;

define <4 x i16> @trunc_v4i64_v4i16(<4 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v4i64_v4i16:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; CHECK-NEXT: uzp1 z0.h, [[A_WORDS]].h, [[A_WORDS]].h
; CHECK-NEXT: ret
  %a = load <4 x i64>, <4 x i64>* %in
  %b = trunc <4 x i64> %a to <4 x i16>
  ret <4 x i16> %b
}

define <8 x i16> @trunc_v8i64_v8i16(<8 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v8i64_v8i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_512-NEXT: uzp1 z0.h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %in
  %b = trunc <8 x i64> %a to <8 x i16>
  ret <8 x i16> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v16i64_v16i16(<16 x i64>* %in, <16 x i16>* %out) #0 {
; CHECK-LABEL: trunc_v16i64_v16i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_1024: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_1024: add [[A_HALFS]].h, [[A_HALFS]].h, [[A_HALFS]].h
  %a = load <16 x i64>, <16 x i64>* %in
  %b = trunc <16 x i64> %a to <16 x i16>
  %c = add <16 x i16> %b, %b
  store <16 x i16> %c, <16 x i16>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i64_v32i16(<32 x i64>* %in, <32 x i16>* %out) #0 {
; CHECK-LABEL: trunc_v32i64_v32i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_2048: uzp1 [[A_HALFS:z[0-9]+]].h, [[A_WORDS]].h, [[A_WORDS]].h
; VBITS_GE_2048: add [[A_HALFS]].h, [[A_HALFS]].h, [[A_HALFS]].h
  %a = load <32 x i64>, <32 x i64>* %in
  %b = trunc <32 x i64> %a to <32 x i16>
  %c = add <32 x i16> %b, %b
  store <32 x i16> %c, <32 x i16>* %out
  ret void
}

;
; truncate i64 -> i32
;

define <4 x i32> @trunc_v4i64_v4i32(<4 x i64>* %in) #0 {
; CHECK-LABEL: trunc_v4i64_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: uzp1 z0.s, [[A_DWORDS]].s, [[A_DWORDS]].s
; CHECK-NEXT: ret
  %a = load <4 x i64>, <4 x i64>* %in
  %b = trunc <4 x i64> %a to <4 x i32>
  ret <4 x i32> %b
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v8i64_v8i32(<8 x i64>* %in, <8 x i32>* %out) #0 {
; CHECK-LABEL: trunc_v8i64_v8i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_512: add [[A_WORDS]].s, [[A_WORDS]].s, [[A_WORDS]].s
  %a = load <8 x i64>, <8 x i64>* %in
  %b = trunc <8 x i64> %a to <8 x i32>
  %c = add <8 x i32> %b, %b
  store <8 x i32> %c, <8 x i32>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v16i64_v16i32(<16 x i64>* %in, <16 x i32>* %out) #0 {
; CHECK-LABEL: trunc_v16i64_v16i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_1024: add [[A_WORDS]].s, [[A_WORDS]].s, [[A_WORDS]].s
  %a = load <16 x i64>, <16 x i64>* %in
  %b = trunc <16 x i64> %a to <16 x i32>
  %c = add <16 x i32> %b, %b
  store <16 x i32> %c, <16 x i32>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the truncate being combined with the store.
define void @trunc_v32i64_v32i32(<32 x i64>* %in, <32 x i32>* %out) #0 {
; CHECK-LABEL: trunc_v32i64_v32i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048: ld1d { [[A_DWORDS:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048: uzp1 [[A_WORDS:z[0-9]+]].s, [[A_DWORDS]].s, [[A_DWORDS]].s
; VBITS_GE_2048: add [[A_WORDS]].s, [[A_WORDS]].s, [[A_WORDS]].s
  %a = load <32 x i64>, <32 x i64>* %in
  %b = trunc <32 x i64> %a to <32 x i32>
  %c = add <32 x i32> %b, %b
  store <32 x i32> %c, <32 x i32>* %out
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
