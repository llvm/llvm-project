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
; ICMP EQ
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @icmp_eq_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: icmp_eq_v8i8:
; CHECK: cmeq v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %cmp = icmp eq <8 x i8> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %sext
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @icmp_eq_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: icmp_eq_v16i8:
; CHECK: cmeq v0.16b, v0.16b, v1.16b
; CHECK-NEXT: ret
  %cmp = icmp eq <16 x i8> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}

define void @icmp_eq_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: icmp_eq_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpeq [[CMP:p[0-9]+]].b, [[PG]]/z, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].b, [[CMP]]/z, #-1
; CHECK-NEXT: st1b { [[SEXT]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %cmp = icmp eq <32 x i8> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i8>
  store <32 x i8> %sext, <32 x i8>* %a
  ret void
}

define void @icmp_eq_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: icmp_eq_v64i8:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[CMP:p[0-9]+]].b, [[PG]]/z, [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].b, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1b { [[SEXT]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[OFF_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: ld1b { [[OP1_LO:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1b { [[OP1_HI:z[0-9]+]].b }, [[PG]]/z, [x0, x[[OFF_HI]]]
; VBITS_EQ_256-DAG: ld1b { [[OP2_LO:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1b { [[OP2_HI:z[0-9]+]].b }, [[PG]]/z, [x1, x[[OFF_HI]]]
; VBITS_EQ_256-DAG: cmpeq [[CMP_LO:p[0-9]+]].b, [[PG]]/z, [[OP1_LO]].b, [[OP2_LO]].b
; VBITS_EQ_256-DAG: cmpeq [[CMP_HI:p[0-9]+]].b, [[PG]]/z, [[OP1_HI]].b, [[OP2_HI]].b
; VBITS_EQ_256-DAG: mov [[SEXT_LO:z[0-9]+]].b, [[CMP_LO]]/z, #-1
; VBITS_EQ_256-DAG: mov [[SEXT_HI:z[0-9]+]].b, [[CMP_HI]]/z, #-1
; VBITS_EQ_256-DAG: st1b { [[SEXT_LO]].b }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1b { [[SEXT_HI]].b }, [[PG]], [x0, x[[OFF_HI]]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %cmp = icmp eq <64 x i8> %op1, %op2
  %sext = sext <64 x i1> %cmp to <64 x i8>
  store <64 x i8> %sext, <64 x i8>* %a
  ret void
}

define void @icmp_eq_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: icmp_eq_v128i8:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].b, [[PG]]/z, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: mov [[SEXT:z[0-9]+]].b, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: st1b { [[SEXT]].b }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %cmp = icmp eq <128 x i8> %op1, %op2
  %sext = sext <128 x i1> %cmp to <128 x i8>
  store <128 x i8> %sext, <128 x i8>* %a
  ret void
}

define void @icmp_eq_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: icmp_eq_v256i8:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].b, [[PG]]/z, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: mov [[SEXT:z[0-9]+]].b, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: st1b { [[SEXT]].b }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %cmp = icmp eq <256 x i8> %op1, %op2
  %sext = sext <256 x i1> %cmp to <256 x i8>
  store <256 x i8> %sext, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @icmp_eq_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: icmp_eq_v4i16:
; CHECK: cmeq v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %cmp = icmp eq <4 x i16> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %sext
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @icmp_eq_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: icmp_eq_v8i16:
; CHECK: cmeq v0.8h, v0.8h, v1.8h
; CHECK-NEXT: ret
  %cmp = icmp eq <8 x i16> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}

define void @icmp_eq_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: icmp_eq_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; CHECK-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %cmp = icmp eq <16 x i16> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %a
  ret void
}

define void @icmp_eq_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: icmp_eq_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1h { [[OP1_LO:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1h { [[OP1_HI:z[0-9]+]].h }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1h { [[OP2_LO:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1h { [[OP2_HI:z[0-9]+]].h }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: cmpeq [[CMP_LO:p[0-9]+]].h, [[PG]]/z, [[OP1_LO]].h, [[OP2_LO]].h
; VBITS_EQ_256-DAG: cmpeq [[CMP_HI:p[0-9]+]].h, [[PG]]/z, [[OP1_HI]].h, [[OP2_HI]].h
; VBITS_EQ_256-DAG: mov [[SEXT_LO:z[0-9]+]].h, [[CMP_LO]]/z, #-1
; VBITS_EQ_256-DAG: mov [[SEXT_HI:z[0-9]+]].h, [[CMP_HI]]/z, #-1
; VBITS_EQ_256-DAG: st1h { [[SEXT_LO]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[SEXT_HI]].h }, [[PG]], [x[[A_HI]]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %cmp = icmp eq <32 x i16> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i16>
  store <32 x i16> %sext, <32 x i16>* %a
  ret void
}

define void @icmp_eq_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: icmp_eq_v64i16:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %cmp = icmp eq <64 x i16> %op1, %op2
  %sext = sext <64 x i1> %cmp to <64 x i16>
  store <64 x i16> %sext, <64 x i16>* %a
  ret void
}

define void @icmp_eq_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: icmp_eq_v128i16:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %cmp = icmp eq <128 x i16> %op1, %op2
  %sext = sext <128 x i1> %cmp to <128 x i16>
  store <128 x i16> %sext, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @icmp_eq_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: icmp_eq_v2i32:
; CHECK: cmeq v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %cmp = icmp eq <2 x i32> %op1, %op2
  %sext = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %sext
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @icmp_eq_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: icmp_eq_v4i32:
; CHECK: cmeq v0.4s, v0.4s, v1.4s
; CHECK-NEXT: ret
  %cmp = icmp eq <4 x i32> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}

define void @icmp_eq_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: icmp_eq_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; CHECK-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %cmp = icmp eq <8 x i32> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i32>
  store <8 x i32> %sext, <8 x i32>* %a
  ret void
}

define void @icmp_eq_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: icmp_eq_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1w { [[OP1_LO:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1w { [[OP1_HI:z[0-9]+]].s }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1w { [[OP2_LO:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1w { [[OP2_HI:z[0-9]+]].s }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: cmpeq [[CMP_LO:p[0-9]+]].s, [[PG]]/z, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-DAG: cmpeq [[CMP_HI:p[0-9]+]].s, [[PG]]/z, [[OP1_HI]].s, [[OP2_HI]].s
; VBITS_EQ_256-DAG: mov [[SEXT_LO:z[0-9]+]].s, [[CMP_LO]]/z, #-1
; VBITS_EQ_256-DAG: mov [[SEXT_HI:z[0-9]+]].s, [[CMP_HI]]/z, #-1
; VBITS_EQ_256-DAG: st1w { [[SEXT_LO]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[SEXT_HI]].s }, [[PG]], [x[[A_HI]]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %cmp = icmp eq <16 x i32> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i32>
  store <16 x i32> %sext, <16 x i32>* %a
  ret void
}

define void @icmp_eq_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: icmp_eq_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %cmp = icmp eq <32 x i32> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i32>
  store <32 x i32> %sext, <32 x i32>* %a
  ret void
}

define void @icmp_eq_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: icmp_eq_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %cmp = icmp eq <64 x i32> %op1, %op2
  %sext = sext <64 x i1> %cmp to <64 x i32>
  store <64 x i32> %sext, <64 x i32>* %a
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @icmp_eq_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: icmp_eq_v1i64:
; CHECK: cmeq d0, d0, d1
; CHECK-NEXT: ret
  %cmp = icmp eq <1 x i64> %op1, %op2
  %sext = sext <1 x i1> %cmp to <1 x i64>
  ret <1 x i64> %sext
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @icmp_eq_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: icmp_eq_v2i64:
; CHECK: cmeq v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret
  %cmp = icmp eq <2 x i64> %op1, %op2
  %sext = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %sext
}

define void @icmp_eq_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: icmp_eq_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpeq [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; CHECK-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %cmp = icmp eq <4 x i64> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i64>
  store <4 x i64> %sext, <4 x i64>* %a
  ret void
}

define void @icmp_eq_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: icmp_eq_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpeq [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[A_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: ld1d { [[OP1_LO:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_EQ_256-DAG: ld1d { [[OP1_HI:z[0-9]+]].d }, [[PG]]/z, [x[[A_HI]]]
; VBITS_EQ_256-DAG: ld1d { [[OP2_LO:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_EQ_256-DAG: ld1d { [[OP2_HI:z[0-9]+]].d }, [[PG]]/z, [x[[B_HI]]]
; VBITS_EQ_256-DAG: cmpeq [[CMP_LO:p[0-9]+]].d, [[PG]]/z, [[OP1_LO]].d, [[OP2_LO]].d
; VBITS_EQ_256-DAG: cmpeq [[CMP_HI:p[0-9]+]].d, [[PG]]/z, [[OP1_HI]].d, [[OP2_HI]].d
; VBITS_EQ_256-DAG: mov [[SEXT_LO:z[0-9]+]].d, [[CMP_LO]]/z, #-1
; VBITS_EQ_256-DAG: mov [[SEXT_HI:z[0-9]+]].d, [[CMP_HI]]/z, #-1
; VBITS_EQ_256-DAG: st1d { [[SEXT_LO]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[SEXT_HI]].d }, [[PG]], [x[[A_HI]]]
; VBITS_EQ_256-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %cmp = icmp eq <8 x i64> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i64>
  store <8 x i64> %sext, <8 x i64>* %a
  ret void
}

define void @icmp_eq_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: icmp_eq_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmpeq [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %cmp = icmp eq <16 x i64> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i64>
  store <16 x i64> %sext, <16 x i64>* %a
  ret void
}

define void @icmp_eq_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: icmp_eq_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: cmpeq [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %cmp = icmp eq <32 x i64> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i64>
  store <32 x i64> %sext, <32 x i64>* %a
  ret void
}

;
; ICMP NE
;

define void @icmp_ne_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: icmp_ne_v32i8:
; CHECK: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-DAG: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; CHECK-DAG: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpne [[CMP:p[0-9]+]].b, [[PG]]/z, [[OP1]].b, [[OP2]].b
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].b, [[CMP]]/z, #-1
; CHECK-NEXT: st1b { [[SEXT]].b }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %cmp = icmp ne <32 x i8> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i8>
  store <32 x i8> %sext, <32 x i8>* %a
  ret void
}

;
; ICMP SGE
;

define void @icmp_sge_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: icmp_sge_v32i16:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpge [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %cmp = icmp sge <32 x i16> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i16>
  store <32 x i16> %sext, <32 x i16>* %a
  ret void
}

;
; ICMP SGT
;

define void @icmp_sgt_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: icmp_sgt_v16i16:
; CHECK: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-DAG: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; CHECK-DAG: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpgt [[CMP:p[0-9]+]].h, [[PG]]/z, [[OP1]].h, [[OP2]].h
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].h, [[CMP]]/z, #-1
; CHECK-NEXT: st1h { [[SEXT]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %cmp = icmp sgt <16 x i16> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i16>
  store <16 x i16> %sext, <16 x i16>* %a
  ret void
}

;
; ICMP SLE
;

define void @icmp_sle_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: icmp_sle_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmpge [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP2]].s, [[OP1]].s
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %cmp = icmp sle <16 x i32> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i32>
  store <16 x i32> %sext, <16 x i32>* %a
  ret void
}

;
; ICMP SLT
;

define void @icmp_slt_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: icmp_slt_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-DAG: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-DAG: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: cmpgt [[CMP:p[0-9]+]].s, [[PG]]/z, [[OP2]].s, [[OP1]].s
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].s, [[CMP]]/z, #-1
; CHECK-NEXT: st1w { [[SEXT]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %cmp = icmp slt <8 x i32> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i32>
  store <8 x i32> %sext, <8 x i32>* %a
  ret void
}

;
; ICMP UGE
;

define void @icmp_uge_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: icmp_uge_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: cmphs [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_512-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %cmp = icmp uge <8 x i64> %op1, %op2
  %sext = sext <8 x i1> %cmp to <8 x i64>
  store <8 x i64> %sext, <8 x i64>* %a
  ret void
}

;
; ICMP UGT
;

define void @icmp_ugt_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: icmp_ugt_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: cmphi [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; CHECK-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %cmp = icmp ugt <4 x i64> %op1, %op2
  %sext = sext <4 x i1> %cmp to <4 x i64>
  store <4 x i64> %sext, <4 x i64>* %a
  ret void
}

;
; ICMP ULE
;

define void @icmp_ule_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: icmp_ule_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: cmphs [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP2]].d, [[OP1]].d
; VBITS_GE_1024-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_1024-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %cmp = icmp ule <16 x i64> %op1, %op2
  %sext = sext <16 x i1> %cmp to <16 x i64>
  store <16 x i64> %sext, <16 x i64>* %a
  ret void
}

;
; ICMP ULT
;

define void @icmp_ult_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: icmp_ult_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-DAG: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-DAG: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: cmphi [[CMP:p[0-9]+]].d, [[PG]]/z, [[OP2]].d, [[OP1]].d
; VBITS_GE_2048-NEXT: mov [[SEXT:z[0-9]+]].d, [[CMP]]/z, #-1
; VBITS_GE_2048-NEXT: st1d { [[SEXT]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %cmp = icmp ult <32 x i64> %op1, %op2
  %sext = sext <32 x i1> %cmp to <32 x i64>
  store <32 x i64> %sext, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
