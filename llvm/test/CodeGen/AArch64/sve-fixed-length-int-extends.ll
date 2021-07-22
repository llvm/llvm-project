; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
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
; sext i1 -> i32
;

; NOTE: Covers the scenario where a SIGN_EXTEND_INREG is required, whose inreg
; type's element type is not byte based and thus cannot be lowered directly to
; an SVE instruction.
define void @sext_v8i1_v8i32(<8 x i1> %a, <8 x i32>* %out) #0 {
; CHECK-LABEL: sext_v8i1_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; CHECK-NEXT: lsl [[A_WORDS]].s, [[PG]]/m, [[A_WORDS]].s, #31
; CHECK-NEXT: asr [[A_WORDS]].s, [[PG]]/m, [[A_WORDS]].s, #31
; CHECK-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <8 x i1> %a to <8 x i32>
  store <8 x i32> %b, <8 x i32>* %out
  ret void
}

;
; sext i3 -> i64
;

; NOTE: Covers the scenario where a SIGN_EXTEND_INREG is required, whose inreg
; type's element type is not power-of-2 based and thus cannot be lowered
; directly to an SVE instruction.
define void @sext_v4i3_v4i64(<4 x i3> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: sext_v4i3_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; CHECK-NEXT: lsl [[A_DWORDS]].d, [[PG]]/m, [[A_DWORDS]].d, #61
; CHECK-NEXT: asr [[A_DWORDS]].d, [[PG]]/m, [[A_DWORDS]].d, #61
; CHECK-NEXT: st1d { [[A_WORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <4 x i3> %a to <4 x i64>
  store <4 x i64> %b, <4 x i64>* %out
  ret void
}

;
; sext i8 -> i16
;

define void @sext_v16i8_v16i16(<16 x i8> %a, <16 x i16>* %out) #0 {
; CHECK-LABEL: sext_v16i8_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; CHECK-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <16 x i8> %a to <16 x i16>
  store <16 x i16>%b, <16 x i16>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the extend being combined with the load.
define void @sext_v32i8_v32i16(<32 x i8>* %in, <32 x i16>* %out) #0 {
; CHECK-LABEL: sext_v32i8_v32i16:
; VBITS_GE_512: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_512-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = sext <32 x i8> %b to <32 x i16>
  store <32 x i16> %c, <32 x i16>* %out
  ret void
}

define void @sext_v64i8_v64i16(<64 x i8>* %in, <64 x i16>* %out) #0 {
; CHECK-LABEL: sext_v64i8_v64i16:
; VBITS_GE_1024: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_1024-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <64 x i8>, <64 x i8>* %in
  %b = add <64 x i8> %a, %a
  %c = sext <64 x i8> %b to <64 x i16>
  store <64 x i16> %c, <64 x i16>* %out
  ret void
}

define void @sext_v128i8_v128i16(<128 x i8>* %in, <128 x i16>* %out) #0 {
; CHECK-LABEL: sext_v128i8_v128i16:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <128 x i8>, <128 x i8>* %in
  %b = add <128 x i8> %a, %a
  %c = sext <128 x i8> %b to <128 x i16>
  store <128 x i16> %c, <128 x i16>* %out
  ret void
}

;
; sext i8 -> i32
;

define void @sext_v8i8_v8i32(<8 x i8> %a, <8 x i32>* %out) #0 {
; CHECK-LABEL: sext_v8i8_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; CHECK-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; CHECK-NEXT: st1w { [[A_HALFS]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <8 x i8> %a to <8 x i32>
  store <8 x i32>%b, <8 x i32>* %out
  ret void
}

define void @sext_v16i8_v16i32(<16 x i8> %a, <16 x i32>* %out) #0 {
; CHECK-LABEL: sext_v16i8_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ext v[[A_HI:[0-9]+]].16b, v0.16b, v0.16b, #8
; VBITS_EQ_256-DAG: sunpklo [[A_HALFS_LO:z[0-9]+]].h, z0.b
; VBITS_EQ_256-DAG: sunpklo [[A_HALFS_HI:z[0-9]+]].h, z[[A_HI]].b
; VBITS_EQ_256-DAG: sunpklo [[A_WORDS_LO:z[0-9]+]].s, [[A_HALFS_LO]].h
; VBITS_EQ_256-DAG: sunpklo [[A_WORDS_HI:z[0-9]+]].s, [[A_HALFS_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[NUMELTS:[0-9]+]], #8
; VBITS_EQ_256-DAG: st1w { [[A_WORDS_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[A_WORDS_HI]].s }, [[PG]], [x0, x[[NUMELTS]], lsl #2]
; VBITS_EQ_256-NEXT: ret
  %b = sext <16 x i8> %a to <16 x i32>
  store <16 x i32> %b, <16 x i32>* %out
  ret void
}

define void @sext_v32i8_v32i32(<32 x i8>* %in, <32 x i32>* %out) #0 {
; CHECK-LABEL: sext_v32i8_v32i32:
; VBITS_GE_1024: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_1024-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_1024-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = sext <32 x i8> %b to <32 x i32>
  store <32 x i32> %c, <32 x i32>* %out
  ret void
}

define void @sext_v64i8_v64i32(<64 x i8>* %in, <64 x i32>* %out) #0 {
; CHECK-LABEL: sext_v64i8_v64i32:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i8>, <64 x i8>* %in
  %b = add <64 x i8> %a, %a
  %c = sext <64 x i8> %b to <64 x i32>
  store <64 x i32> %c, <64 x i32>* %out
  ret void
}

;
; sext i8 -> i64
;

; NOTE: v4i8 is an unpacked typed stored within a v4i16 container. The sign
; extend is a two step process where the container is any_extend'd with the
; result feeding an inreg sign extend.
define void @sext_v4i8_v4i64(<4 x i8> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: sext_v4i8_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[ANYEXT_W:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[ANYEXT_D:z[0-9]+]].d, [[ANYEXT_W]].s
; CHECK-NEXT: sxtb [[A_DWORDS:z[0-9]+]].d, [[PG]]/m, [[ANYEXT_D]].d
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <4 x i8> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @sext_v8i8_v8i64(<8 x i8> %a, <8 x i64>* %out) #0 {
; CHECK-LABEL: sext_v8i8_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %b = sext <8 x i8> %a to <8 x i64>
  store <8 x i64>%b, <8 x i64>* %out
  ret void
}

define void @sext_v16i8_v16i64(<16 x i8> %a, <16 x i64>* %out) #0 {
; CHECK-LABEL: sext_v16i8_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_1024-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %b = sext <16 x i8> %a to <16 x i64>
  store <16 x i64> %b, <16 x i64>* %out
  ret void
}

define void @sext_v32i8_v32i64(<32 x i8>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: sext_v32i8_v32i64:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: sunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = sext <32 x i8> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

;
; sext i16 -> i32
;

define void @sext_v8i16_v8i32(<8 x i16> %a, <8 x i32>* %out) #0 {
; CHECK-LABEL: sext_v8i16_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <8 x i16> %a to <8 x i32>
  store <8 x i32>%b, <8 x i32>* %out
  ret void
}

define void @sext_v16i16_v16i32(<16 x i16>* %in, <16 x i32>* %out) #0 {
; CHECK-LABEL: sext_v16i16_v16i32:
; VBITS_GE_512: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_512-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %in
  %b = add <16 x i16> %a, %a
  %c = sext <16 x i16> %b to <16 x i32>
  store <16 x i32> %c, <16 x i32>* %out
  ret void
}

define void @sext_v32i16_v32i32(<32 x i16>* %in, <32 x i32>* %out) #0 {
; CHECK-LABEL: sext_v32i16_v32i32:
; VBITS_GE_1024: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_1024-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %in
  %b = add <32 x i16> %a, %a
  %c = sext <32 x i16> %b to <32 x i32>
  store <32 x i32> %c, <32 x i32>* %out
  ret void
}

define void @sext_v64i16_v64i32(<64 x i16>* %in, <64 x i32>* %out) #0 {
; CHECK-LABEL: sext_v64i16_v64i32:
; VBITS_GE_2048: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_2048-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i16>, <64 x i16>* %in
  %b = add <64 x i16> %a, %a
  %c = sext <64 x i16> %b to <64 x i32>
  store <64 x i32> %c, <64 x i32>* %out
  ret void
}

;
; sext i16 -> i64
;

define void @sext_v4i16_v4i64(<4 x i16> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: sext_v4i16_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <4 x i16> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @sext_v8i16_v8i64(<8 x i16> %a, <8 x i64>* %out) #0 {
; CHECK-LABEL: sext_v8i16_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; VBITS_GE_512-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %b = sext <8 x i16> %a to <8 x i64>
  store <8 x i64>%b, <8 x i64>* %out
  ret void
}

define void @sext_v16i16_v16i64(<16 x i16>* %in, <16 x i64>* %out) #0 {
; CHECK-LABEL: sext_v16i16_v16i64:
; VBITS_GE_1024: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_1024-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %in
  %b = add <16 x i16> %a, %a
  %c = sext <16 x i16> %b to <16 x i64>
  store <16 x i64> %c, <16 x i64>* %out
  ret void
}

define void @sext_v32i16_v32i64(<32 x i16>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: sext_v32i16_v32i64:
; VBITS_GE_2048: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_2048-NEXT: sunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %in
  %b = add <32 x i16> %a, %a
  %c = sext <32 x i16> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

;
; sext i32 -> i64
;

define void @sext_v4i32_v4i64(<4 x i32> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: sext_v4i32_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, z0.s
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = sext <4 x i32> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @sext_v8i32_v8i64(<8 x i32>* %in, <8 x i64>* %out) #0 {
; CHECK-LABEL: sext_v8i32_v8i64:
; VBITS_GE_512: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_512-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %in
  %b = add <8 x i32> %a, %a
  %c = sext <8 x i32> %b to <8 x i64>
  store <8 x i64> %c, <8 x i64>* %out
  ret void
}

define void @sext_v16i32_v16i64(<16 x i32>* %in, <16 x i64>* %out) #0 {
; CHECK-LABEL: sext_v16i32_v16i64:
; VBITS_GE_1024: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_1024-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %in
  %b = add <16 x i32> %a, %a
  %c = sext <16 x i32> %b to <16 x i64>
  store <16 x i64> %c, <16 x i64>* %out
  ret void
}

define void @sext_v32i32_v32i64(<32 x i32>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: sext_v32i32_v32i64:
; VBITS_GE_2048: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_2048-NEXT: sunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i32>, <32 x i32>* %in
  %b = add <32 x i32> %a, %a
  %c = sext <32 x i32> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

;
; zext i8 -> i16
;

define void @zext_v16i8_v16i16(<16 x i8> %a, <16 x i16>* %out) #0 {
; CHECK-LABEL: zext_v16i8_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; CHECK-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <16 x i8> %a to <16 x i16>
  store <16 x i16>%b, <16 x i16>* %out
  ret void
}

; NOTE: Extra 'add' is to prevent the extend being combined with the load.
define void @zext_v32i8_v32i16(<32 x i8>* %in, <32 x i16>* %out) #0 {
; CHECK-LABEL: zext_v32i8_v32i16:
; VBITS_GE_512: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_512-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = zext <32 x i8> %b to <32 x i16>
  store <32 x i16> %c, <32 x i16>* %out
  ret void
}

define void @zext_v64i8_v64i16(<64 x i8>* %in, <64 x i16>* %out) #0 {
; CHECK-LABEL: zext_v64i8_v64i16:
; VBITS_GE_1024: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_1024-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <64 x i8>, <64 x i8>* %in
  %b = add <64 x i8> %a, %a
  %c = zext <64 x i8> %b to <64 x i16>
  store <64 x i16> %c, <64 x i16>* %out
  ret void
}

define void @zext_v128i8_v128i16(<128 x i8>* %in, <128 x i16>* %out) #0 {
; CHECK-LABEL: zext_v128i8_v128i16:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[A_HALFS]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <128 x i8>, <128 x i8>* %in
  %b = add <128 x i8> %a, %a
  %c = zext <128 x i8> %b to <128 x i16>
  store <128 x i16> %c, <128 x i16>* %out
  ret void
}

;
; zext i8 -> i32
;

define void @zext_v8i8_v8i32(<8 x i8> %a, <8 x i32>* %out) #0 {
; CHECK-LABEL: zext_v8i8_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; CHECK-NEXT: st1w { [[A_HALFS]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <8 x i8> %a to <8 x i32>
  store <8 x i32>%b, <8 x i32>* %out
  ret void
}

define void @zext_v16i8_v16i32(<16 x i8> %a, <16 x i32>* %out) #0 {
; CHECK-LABEL: zext_v16i8_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: ext v[[A_HI:[0-9]+]].16b, v0.16b, v0.16b, #8
; VBITS_EQ_256-DAG: uunpklo [[A_HALFS_LO:z[0-9]+]].h, z0.b
; VBITS_EQ_256-DAG: uunpklo [[A_HALFS_HI:z[0-9]+]].h, z[[A_HI]].b
; VBITS_EQ_256-DAG: uunpklo [[A_WORDS_LO:z[0-9]+]].s, [[A_HALFS_LO]].h
; VBITS_EQ_256-DAG: uunpklo [[A_WORDS_HI:z[0-9]+]].s, [[A_HALFS_HI]].h
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: mov x[[OUT_HI:[0-9]+]], #8
; VBITS_EQ_256-DAG: st1w { [[A_WORDS_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[A_WORDS_HI]].s }, [[PG]], [x0, x[[OUT_HI]], lsl #2]
; VBITS_EQ_256-NEXT: ret
  %b = zext <16 x i8> %a to <16 x i32>
  store <16 x i32> %b, <16 x i32>* %out
  ret void
}

define void @zext_v32i8_v32i32(<32 x i8>* %in, <32 x i32>* %out) #0 {
; CHECK-LABEL: zext_v32i8_v32i32:
; VBITS_GE_1024: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_1024-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_1024-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = zext <32 x i8> %b to <32 x i32>
  store <32 x i32> %c, <32 x i32>* %out
  ret void
}

define void @zext_v64i8_v64i32(<64 x i8>* %in, <64 x i32>* %out) #0 {
; CHECK-LABEL: zext_v64i8_v64i32:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i8>, <64 x i8>* %in
  %b = add <64 x i8> %a, %a
  %c = zext <64 x i8> %b to <64 x i32>
  store <64 x i32> %c, <64 x i32>* %out
  ret void
}

;
; zext i8 -> i64
;

; NOTE: v4i8 is an unpacked typed stored within a v4i16 container. The zero
; extend is a two step process where the container is zero_extend_inreg'd with
; the result feeding a normal zero extend from halfs to doublewords.
define void @zext_v4i8_v4i64(<4 x i8> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: zext_v4i8_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: bic v0.4h, #255, lsl #8
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <4 x i8> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @zext_v8i8_v8i64(<8 x i8> %a, <8 x i64>* %out) #0 {
; CHECK-LABEL: zext_v8i8_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %b = zext <8 x i8> %a to <8 x i64>
  store <8 x i64>%b, <8 x i64>* %out
  ret void
}

define void @zext_v16i8_v16i64(<16 x i8> %a, <16 x i64>* %out) #0 {
; CHECK-LABEL: zext_v16i8_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, z0.b
; VBITS_GE_1024-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %b = zext <16 x i8> %a to <16 x i64>
  store <16 x i64> %b, <16 x i64>* %out
  ret void
}

define void @zext_v32i8_v32i64(<32 x i8>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: zext_v32i8_v32i64:
; VBITS_GE_2048: add [[A_BYTES:z[0-9]+]].b, {{p[0-9]+}}/m, {{z[0-9]+}}.b, {{z[0-9]+}}.b
; VBITS_GE_2048-NEXT: uunpklo [[A_HALFS:z[0-9]+]].h, [[A_BYTES]].b
; VBITS_GE_2048-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i8>, <32 x i8>* %in
  %b = add <32 x i8> %a, %a
  %c = zext <32 x i8> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

;
; zext i16 -> i32
;

define void @zext_v8i16_v8i32(<8 x i16> %a, <8 x i32>* %out) #0 {
; CHECK-LABEL: zext_v8i16_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <8 x i16> %a to <8 x i32>
  store <8 x i32>%b, <8 x i32>* %out
  ret void
}

define void @zext_v16i16_v16i32(<16 x i16>* %in, <16 x i32>* %out) #0 {
; CHECK-LABEL: zext_v16i16_v16i32:
; VBITS_GE_512: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_512-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %in
  %b = add <16 x i16> %a, %a
  %c = zext <16 x i16> %b to <16 x i32>
  store <16 x i32> %c, <16 x i32>* %out
  ret void
}

define void @zext_v32i16_v32i32(<32 x i16>* %in, <32 x i32>* %out) #0 {
; CHECK-LABEL: zext_v32i16_v32i32:
; VBITS_GE_1024: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_1024-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %in
  %b = add <32 x i16> %a, %a
  %c = zext <32 x i16> %b to <32 x i32>
  store <32 x i32> %c, <32 x i32>* %out
  ret void
}

define void @zext_v64i16_v64i32(<64 x i16>* %in, <64 x i32>* %out) #0 {
; CHECK-LABEL: zext_v64i16_v64i32:
; VBITS_GE_2048: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_2048-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[A_WORDS]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <64 x i16>, <64 x i16>* %in
  %b = add <64 x i16> %a, %a
  %c = zext <64 x i16> %b to <64 x i32>
  store <64 x i32> %c, <64 x i32>* %out
  ret void
}

;
; zext i16 -> i64
;

define void @zext_v4i16_v4i64(<4 x i16> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: zext_v4i16_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; CHECK-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <4 x i16> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @zext_v8i16_v8i64(<8 x i16> %a, <8 x i64>* %out) #0 {
; CHECK-LABEL: zext_v8i16_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, z0.h
; VBITS_GE_512-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %b = zext <8 x i16> %a to <8 x i64>
  store <8 x i64>%b, <8 x i64>* %out
  ret void
}

define void @zext_v16i16_v16i64(<16 x i16>* %in, <16 x i64>* %out) #0 {
; CHECK-LABEL: zext_v16i16_v16i64:
; VBITS_GE_1024: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_1024-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_1024-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i16>, <16 x i16>* %in
  %b = add <16 x i16> %a, %a
  %c = zext <16 x i16> %b to <16 x i64>
  store <16 x i64> %c, <16 x i64>* %out
  ret void
}

define void @zext_v32i16_v32i64(<32 x i16>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: zext_v32i16_v32i64:
; VBITS_GE_2048: add [[A_HALFS:z[0-9]+]].h, {{p[0-9]+}}/m, {{z[0-9]+}}.h, {{z[0-9]+}}.h
; VBITS_GE_2048-NEXT: uunpklo [[A_WORDS:z[0-9]+]].s, [[A_HALFS]].h
; VBITS_GE_2048-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %in
  %b = add <32 x i16> %a, %a
  %c = zext <32 x i16> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

;
; zext i32 -> i64
;

define void @zext_v4i32_v4i64(<4 x i32> %a, <4 x i64>* %out) #0 {
; CHECK-LABEL: zext_v4i32_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, z0.s
; CHECK-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %b = zext <4 x i32> %a to <4 x i64>
  store <4 x i64>%b, <4 x i64>* %out
  ret void
}

define void @zext_v8i32_v8i64(<8 x i32>* %in, <8 x i64>* %out) #0 {
; CHECK-LABEL: zext_v8i32_v8i64:
; VBITS_GE_512: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_512-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_512-NEXT: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i32>, <8 x i32>* %in
  %b = add <8 x i32> %a, %a
  %c = zext <8 x i32> %b to <8 x i64>
  store <8 x i64> %c, <8 x i64>* %out
  ret void
}

define void @zext_v16i32_v16i64(<16 x i32>* %in, <16 x i64>* %out) #0 {
; CHECK-LABEL: zext_v16i32_v16i64:
; VBITS_GE_1024: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_1024-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_1024-NEXT: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %in
  %b = add <16 x i32> %a, %a
  %c = zext <16 x i32> %b to <16 x i64>
  store <16 x i64> %c, <16 x i64>* %out
  ret void
}

define void @zext_v32i32_v32i64(<32 x i32>* %in, <32 x i64>* %out) #0 {
; CHECK-LABEL: zext_v32i32_v32i64:
; VBITS_GE_2048: add [[A_WORDS:z[0-9]+]].s, {{p[0-9]+}}/m, {{z[0-9]+}}.s, {{z[0-9]+}}.s
; VBITS_GE_2048-NEXT: uunpklo [[A_DWORDS:z[0-9]+]].d, [[A_WORDS]].s
; VBITS_GE_2048-NEXT: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[A_DWORDS]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %a = load <32 x i32>, <32 x i32>* %in
  %b = add <32 x i32> %a, %a
  %c = zext <32 x i32> %b to <32 x i64>
  store <32 x i64> %c, <32 x i64>* %out
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }
