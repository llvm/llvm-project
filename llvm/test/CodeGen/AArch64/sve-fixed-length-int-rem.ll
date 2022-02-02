; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=VBITS_EQ_128
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048,VBITS_EQ_2048

target triple = "aarch64-unknown-linux-gnu"

;
; SREM
;

; Vector vXi8 sdiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for >= 256 bits here.
define <8 x i8> @srem_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: srem_v8i8:
; CHECK: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2:z[0-9]+]].b
; CHECK-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1:z[0-9]+]].b
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; CHECK-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; CHECK-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK-NEXT: umov [[SCALAR1:w[0-9]+]], [[VEC:v[0-9]+]].h[0]
; CHECK-NEXT: umov [[SCALAR2:w[0-9]+]], [[VEC]].h[1]
; CHECK-NEXT: fmov s3, [[SCALAR1]]
; CHECK-NEXT: umov [[SCALAR3:w[0-9]+]], [[VEC]].h[2]
; CHECK-NEXT: mov [[FINAL:v[0-9]+]].b[1], [[SCALAR2]]
; CHECK-NEXT: mov [[FINAL]].b[2], [[SCALAR3]]
; CHECK-NEXT: umov [[SCALAR4:w[0-9]+]], [[VEC]].h[3]
; CHECK-NEXT: mov [[FINAL]].b[3], [[SCALAR4]]
; CHECK-NEXT: umov [[SCALAR5:w[0-9]+]], [[VEC]].h[4]
; CHECK-NEXT: mov [[FINAL]].b[4], [[SCALAR5]]
; CHECK-NEXT: umov [[SCALAR6:w[0-9]+]], [[VEC]].h[5]
; CHECK-NEXT: mov [[FINAL]].b[5], [[SCALAR6]]
; CHECK-NEXT: umov [[SCALAR7:w[0-9]+]], [[VEC]].h[6]
; CHECK-NEXT: mov [[FINAL]].b[6], [[SCALAR7]]
; CHECK-NEXT: umov [[SCALAR8:w[0-9]+]], [[VEC]].h[7]
; CHECK-NEXT: mov [[FINAL]].b[7], [[SCALAR8]]
; CHECK-NEXT: mls v0.8b, [[FINAL]].8b, v1.8b
; CHECK: ret

; VBITS_EQ_128-LABEL: srem_v8i8:
; VBITS_EQ_128:         sshll v2.8h, v1.8b, #0
; VBITS_EQ_128-NEXT:    sshll v3.8h, v0.8b, #0
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    sunpkhi z4.s, z2.h
; VBITS_EQ_128-NEXT:    sunpkhi z5.s, z3.h
; VBITS_EQ_128-NEXT:    sunpklo z2.s, z2.h
; VBITS_EQ_128-NEXT:    sunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    sdivr z4.s, p0/m, z4.s, z5.s
; VBITS_EQ_128-NEXT:    sdivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z2.h, z4.h
; VBITS_EQ_128-NEXT:    xtn v2.8b, v2.8h
; VBITS_EQ_128-NEXT:    mls v0.8b, v2.8b, v1.8b
; VBITS_EQ_128-NEXT:    ret

  %res = srem <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

define <16 x i8> @srem_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: srem_v16i8:

; HALF VECTOR
; VBITS_EQ_256: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_256-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: mls v0.16b, v2.16b, v1.16b

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_512: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_512-NEXT: mls v0.16b, v2.16b, v1.16b
; CHECK: ret

; VBITS_EQ_128-LABEL: srem_v16i8:
; VBITS_EQ_128:         sunpkhi z2.h, z1.b
; VBITS_EQ_128-NEXT:    sunpkhi z3.h, z0.b
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    sunpkhi z5.s, z2.h
; VBITS_EQ_128-NEXT:    sunpkhi z6.s, z3.h
; VBITS_EQ_128-NEXT:    sunpklo z2.s, z2.h
; VBITS_EQ_128-NEXT:    sunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    sunpklo z4.h, z1.b
; VBITS_EQ_128-NEXT:    sdivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    sunpklo z3.h, z0.b
; VBITS_EQ_128-NEXT:    sdivr z5.s, p0/m, z5.s, z6.s
; VBITS_EQ_128-NEXT:    sunpkhi z6.s, z4.h
; VBITS_EQ_128-NEXT:    sunpkhi z7.s, z3.h
; VBITS_EQ_128-NEXT:    sunpklo z4.s, z4.h
; VBITS_EQ_128-NEXT:    sunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    sdivr z6.s, p0/m, z6.s, z7.s
; VBITS_EQ_128-NEXT:    sdiv z3.s, p0/m, z3.s, z4.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z2.h, z5.h
; VBITS_EQ_128-NEXT:    uzp1 z3.h, z3.h, z6.h
; VBITS_EQ_128-NEXT:    uzp1 z2.b, z3.b, z2.b
; VBITS_EQ_128-NEXT:    mls v0.16b, v2.16b, v1.16b
; VBITS_EQ_128-NEXT:    ret

  %res = srem <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @srem_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: srem_v32i8:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_256-NEXT: sdivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_256-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_256-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_512-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_512-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_GE_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = srem <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @srem_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: srem_v64i8:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_512-NEXT: sdivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_512-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_1024-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_2048-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_GE_2048-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = srem <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @srem_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: srem_v128i8:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_1024-NEXT: sdivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_2048-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_2048-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_2048-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = srem <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @srem_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: srem_v256i8:

; FULL VECTOR:
; VBITS_EQ_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_EQ_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: sdivr   [[RES_HI_HI:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sdivr   [[RES_HI_LO:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sdiv    [[RES_LO_HI:z[0-9]+]].s, [[PG1]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_2048-NEXT: sdivr   [[RES_LO_LO:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1    [[RES_HI:z[0-9]+]].h, [[RES_HI_LO]].h, [[RES_HI_HI]].h
; VBITS_EQ_2048-NEXT: uzp1    [[RES_LO:z[0-9]+]].h, [[RES_LO_LO]].h, [[RES_LO_HI]].h
; VBITS_EQ_2048-NEXT: uzp1    [[ZIP:z[0-9]+]].b, [[RES_LO]].b, [[RES_HI]].b
; VBITS_EQ_2048-NEXT: mul     [[MUL:z[0-9]+]].b, [[PG]]/m, [[OP2]].b, [[ZIP]].b
; VBITS_EQ_2048-NEXT: sub     [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[MUL]].b
; VBITS_EQ_2048-NEXT: st1b    { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = srem <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Vector vXi16 sdiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for >= 256 bits here.
define <4 x i16> @srem_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: srem_v4i16:
; CHECK: sshll v2.4s, v1.4h, #0
; CHECK-NEXT: sshll v3.4s, v0.4h, #0
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, z2.s, z3.s
; CHECK-NEXT: mov [[SCALAR1:w[0-9]+]], [[VEC:v[0-9]+]].s[1]
; CHECK-NEXT: mov [[SCALAR2:w[0-9]+]], [[VEC]].s[2]
; CHECK-NEXT: mov [[VEC2:v[0-9]+]].16b, [[VEC]].16b
; CHECK-NEXT: mov [[VEC2]].h[1], [[SCALAR1]]
; CHECK-NEXT: mov [[SCALAR3:w[0-9]+]], [[VEC]].s[3]
; CHECK-NEXT: mov [[VEC2]].h[2], [[SCALAR2]]
; CHECK-NEXT: mov [[VEC2]].h[3], [[SCALAR3]]
; CHECK-NEXT: mls v0.4h, [[VEC2]].4h, v1.4h
; CHECK: ret

; VBITS_EQ_128-LABEL: srem_v4i16:
; VBITS_EQ_128:         sshll v2.4s, v1.4h, #0
; VBITS_EQ_128-NEXT:    sshll v3.4s, v0.4h, #0
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    sdivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    xtn v2.4h, v2.4s
; VBITS_EQ_128-NEXT:    mls v0.4h, v2.4h, v1.4h
; VBITS_EQ_128-NEXT:    ret

  %res = srem <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

define <8 x i16> @srem_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: srem_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; CHECK-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; CHECK-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK-NEXT: mls v0.8h, v2.8h, v1.8h
; CHECK: ret

; VBITS_EQ_128-LABEL: srem_v8i16:
; VBITS_EQ_128:         ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    sunpkhi z2.s, z1.h
; VBITS_EQ_128-NEXT:    sunpkhi z3.s, z0.h
; VBITS_EQ_128-NEXT:    sunpklo z4.s, z1.h
; VBITS_EQ_128-NEXT:    sdivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    sunpklo z5.s, z0.h
; VBITS_EQ_128-NEXT:    movprfx z3, z5
; VBITS_EQ_128-NEXT:    sdiv z3.s, p0/m, z3.s, z4.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z3.h, z2.h
; VBITS_EQ_128-NEXT:    mls v0.8h, v2.8h, v1.8h
; VBITS_EQ_128-NEXT:    ret

  %res = srem <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @srem_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: srem_v16i16:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_256-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_256-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_256-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_512-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret

  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = srem <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @srem_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: srem_v32i16:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_512-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_512-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_512-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_1024-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = srem <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @srem_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: srem_v64i16:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_1024-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_1024-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_1024-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_2048-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = srem <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @srem_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: srem_v128i16:
; VBITS_EQ_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: sdivr   [[RES_HI:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: movprfx [[OP3_LO:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_2048-NEXT: sdiv    [[RES_LO:z[0-9]+]].s, [[PG1]]/m, [[OP3_LO]].s, [[OP2_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[ZIP:z[0-9]+]].h, [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_2048-NEXT: mul [[MUL:z[0-9]+]].h, [[PG]]/m, [[OP2]].h, [[ZIP]].h
; VBITS_EQ_2048-NEXT: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[MUL]].h
; VBITS_EQ_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = srem <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector v2i32 sdiv are not legal for NEON so use SVE when available.
define <2 x i32> @srem_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: srem_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], z0
; CHECK-NEXT: sdiv z2.s, [[PG]]/m, [[PFX]].s, z1.s
; CHECK-NEXT: mls v0.2s, v2.2s, v1.2s
; CHECK: ret

; VBITS_EQ_128-LABEL: srem_v2i32:
; VBITS_EQ_128:         ptrue p0.s, vl2
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    sdiv z2.s, p0/m, z2.s, z1.s
; VBITS_EQ_128-NEXT:    mls v0.2s, v2.2s, v1.2s
; VBITS_EQ_128-NEXT:    ret

  %res = srem <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Vector v4i32 sdiv are not legal for NEON so use SVE when available.
define <4 x i32> @srem_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: srem_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], z0
; CHECK-NEXT: sdiv z2.s, [[PG]]/m, [[PFX]].s, z1.s
; CHECK-NEXT: mls v0.4s, v2.4s, v1.4s
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: srem_v4i32:
; VBITS_EQ_128:         ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    sdiv z2.s, p0/m, z2.s, z1.s
; VBITS_EQ_128-NEXT:    mls v0.4s, v2.4s, v1.4s
; VBITS_EQ_128-NEXT:    ret

  %res = srem <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @srem_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: srem_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; CHECK-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; CHECK-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = srem <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @srem_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: srem_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_512-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_512-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_512-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = srem <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @srem_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: srem_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_1024-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_1024-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = srem <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @srem_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: srem_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_2048-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_2048-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = srem <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 sdiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for the 128 bits case here.
define <1 x i64> @srem_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: srem_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, z1.d
; CHECK-NEXT: mul z1.d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub d0, d0, d1
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: srem_v1i64:
; VBITS_EQ_128:         ptrue p0.d, vl1
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    sdiv z2.d, p0/m, z2.d, z1.d
; VBITS_EQ_128-NEXT:    fmov x8, d2
; VBITS_EQ_128-NEXT:    fmov x9, d1
; VBITS_EQ_128-NEXT:    mul x8, x8, x9
; VBITS_EQ_128-NEXT:    fmov d1, x8
; VBITS_EQ_128-NEXT:    sub d0, d0, d1
; VBITS_EQ_128-NEXT:    ret

  %res = srem <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Vector i64 sdiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for the 128 bits case here.
define <2 x i64> @srem_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: srem_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, z1.d
; CHECK-NEXT: mul z1.d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: srem_v2i64:
; VBITS_EQ_128:         ptrue p0.d, vl2
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    sdiv z2.d, p0/m, z2.d, z1.d
; VBITS_EQ_128-NEXT:    fmov x9, d2
; VBITS_EQ_128-NEXT:    fmov x10, d1
; VBITS_EQ_128-NEXT:    mov x8, v2.d[1]
; VBITS_EQ_128-NEXT:    mov x11, v1.d[1]
; VBITS_EQ_128-NEXT:    mul x9, x9, x10
; VBITS_EQ_128-NEXT:    mul x8, x8, x11
; VBITS_EQ_128-NEXT:    fmov d1, x9
; VBITS_EQ_128-NEXT:    mov v1.d[1], x8
; VBITS_EQ_128-NEXT:    sub v0.2d, v0.2d, v1.2d
; VBITS_EQ_128-NEXT:    ret

  %res = srem <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @srem_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: srem_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; CHECK-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = srem <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @srem_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: srem_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_512-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_512-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_512-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = srem <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @srem_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: srem_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_1024-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_1024-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = srem <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @srem_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: srem_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_2048-NEXT: sdiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_2048-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = srem <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UREM
;

; Vector vXi8 udiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for >= 256 bits here.
define <8 x i8> @urem_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: urem_v8i8:
; CHECK: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2:z[0-9]+]].b
; CHECK-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1:z[0-9]+]].b
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; CHECK-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; CHECK-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK-NEXT: umov [[SCALAR0:w[0-9]+]], [[VEC:v[0-9]+]].h[0]
; CHECK-NEXT: umov [[SCALAR1:w[0-9]+]], [[VEC]].h[1]
; CHECK-NEXT: fmov s3, [[SCALAR0]]
; CHECK-NEXT: umov [[SCALAR2:w[0-9]+]], [[VEC]].h[2]
; CHECK-NEXT: mov [[FINAL:v[0-9]+]].b[1], [[SCALAR1]]
; CHECK-NEXT: mov [[FINAL]].b[2], [[SCALAR2]]
; CHECK-NEXT: umov [[SCALAR3:w[0-9]+]], [[VEC]].h[3]
; CHECK-NEXT: mov [[FINAL]].b[3], [[SCALAR3]]
; CHECK-NEXT: umov [[SCALAR4:w[0-9]+]], [[VEC]].h[4]
; CHECK-NEXT: mov [[FINAL]].b[4], [[SCALAR4]]
; CHECK-NEXT: umov [[SCALAR5:w[0-9]+]], [[VEC]].h[5]
; CHECK-NEXT: mov [[FINAL]].b[5], [[SCALAR5]]
; CHECK-NEXT: umov [[SCALAR6:w[0-9]+]], [[VEC]].h[6]
; CHECK-NEXT: mov [[FINAL]].b[6], [[SCALAR6]]
; CHECK-NEXT: umov [[SCALAR7:w[0-9]+]], [[VEC]].h[7]
; CHECK-NEXT: mov [[FINAL]].b[7], [[SCALAR7]]
; CHECK-NEXT: mls v0.8b, [[FINAL]].8b, v1.8b
; CHECK: ret

; VBITS_EQ_128-LABEL: urem_v8i8:
; VBITS_EQ_128:         ushll v2.8h, v1.8b, #0
; VBITS_EQ_128-NEXT:    ushll v3.8h, v0.8b, #0
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    uunpkhi z4.s, z2.h
; VBITS_EQ_128-NEXT:    uunpkhi z5.s, z3.h
; VBITS_EQ_128-NEXT:    uunpklo z2.s, z2.h
; VBITS_EQ_128-NEXT:    uunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    udivr z4.s, p0/m, z4.s, z5.s
; VBITS_EQ_128-NEXT:    udivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z2.h, z4.h
; VBITS_EQ_128-NEXT:    xtn v2.8b, v2.8h
; VBITS_EQ_128-NEXT:    mls v0.8b, v2.8b, v1.8b
; VBITS_EQ_128-NEXT:    ret

  %res = urem <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

define <16 x i8> @urem_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: urem_v16i8:

; HALF VECTOR
; VBITS_EQ_256: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_256-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: mls v0.16b, v2.16b, v1.16b

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_512: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_512-NEXT: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_512-NEXT: mls v0.16b, v2.16b, v1.16b
; CHECK: ret

; VBITS_EQ_128-LABEL: urem_v16i8:
; VBITS_EQ_128:         uunpkhi z2.h, z1.b
; VBITS_EQ_128-NEXT:    uunpkhi z3.h, z0.b
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    uunpkhi z5.s, z2.h
; VBITS_EQ_128-NEXT:    uunpkhi z6.s, z3.h
; VBITS_EQ_128-NEXT:    uunpklo z2.s, z2.h
; VBITS_EQ_128-NEXT:    uunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    uunpklo z4.h, z1.b
; VBITS_EQ_128-NEXT:    udivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    uunpklo z3.h, z0.b
; VBITS_EQ_128-NEXT:    udivr z5.s, p0/m, z5.s, z6.s
; VBITS_EQ_128-NEXT:    uunpkhi z6.s, z4.h
; VBITS_EQ_128-NEXT:    uunpkhi z7.s, z3.h
; VBITS_EQ_128-NEXT:    uunpklo z4.s, z4.h
; VBITS_EQ_128-NEXT:    uunpklo z3.s, z3.h
; VBITS_EQ_128-NEXT:    udivr z6.s, p0/m, z6.s, z7.s
; VBITS_EQ_128-NEXT:    udiv z3.s, p0/m, z3.s, z4.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z2.h, z5.h
; VBITS_EQ_128-NEXT:    uzp1 z3.h, z3.h, z6.h
; VBITS_EQ_128-NEXT:    uzp1 z2.b, z3.b, z2.b
; VBITS_EQ_128-NEXT:    mls v0.16b, v2.16b, v1.16b
; VBITS_EQ_128-NEXT:    ret

  %res = urem <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @urem_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: urem_v32i8:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_256-NEXT: udivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_256-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_256-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_512-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_512-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_GE_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = urem <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @urem_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: urem_v64i8:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_512-NEXT: udivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_512-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_512-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_1024-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_2048-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_2048-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_GE_2048-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = urem <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @urem_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: urem_v128i8:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_1024-NEXT: udivr [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP3]].b
; VBITS_EQ_1024-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_1024-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_2048-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: mul [[OP2]].b, [[PG1]]/m, [[OP2]].b, [[UZP2]].b
; VBITS_EQ_2048-NEXT: sub [[OP1]].b, [[PG1]]/m, [[OP1]].b, [[OP2]].b
; VBITS_EQ_2048-NEXT: st1b { [[OP1:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = urem <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @urem_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: urem_v256i8:
; VBITS_EQ_2048: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_EQ_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: udivr   [[RES_HI_HI:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: udivr   [[RES_HI_LO:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: udiv    [[RES_LO_HI:z[0-9]+]].s, [[PG1]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_2048-NEXT: udivr   [[RES_LO_LO:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO_LO]].s, [[OP1_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1    [[RES_HI:z[0-9]+]].h, [[RES_HI_LO]].h, [[RES_HI_HI]].h
; VBITS_EQ_2048-NEXT: uzp1    [[RES_LO:z[0-9]+]].h, [[RES_LO_LO]].h, [[RES_LO_HI]].h
; VBITS_EQ_2048-NEXT: uzp1    [[ZIP:z[0-9]+]].b, [[RES_LO]].b, [[RES_HI]].b
; VBITS_EQ_2048-NEXT: mul     [[MUL:z[0-9]+]].b, [[PG]]/m, [[OP2]].b, [[ZIP]].b
; VBITS_EQ_2048-NEXT: sub     [[RES:z[0-9]+]].b, [[PG]]/m, [[OP1]].b, [[MUL]].b
; VBITS_EQ_2048-NEXT: st1b    { [[RES]].b }, [[PG]], [x0]
; VBITS_EQ_2048-NEXT: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = urem <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Vector vXi16 udiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for >= 256 bits here.
define <4 x i16> @urem_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: urem_v4i16:
; CHECK: ushll v2.4s, v1.4h, #0
; CHECK-NEXT: ushll v3.4s, v0.4h, #0
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, z2.s, z3.s
; CHECK-NEXT: mov [[SCALAR1:w[0-9]+]], [[VEC:v[0-9]+]].s[1]
; CHECK-NEXT: mov [[SCALAR2:w[0-9]+]], [[VEC]].s[2]
; CHECK-NEXT: mov v3.16b, v2.16b
; CHECK-NEXT: mov [[VECO:v[0-9]+]].h[1], [[SCALAR1]]
; CHECK-NEXT: mov [[SCALAR3:w[0-9]+]], [[VEC]].s[3]
; CHECK-NEXT: mov [[VECO]].h[2], [[SCALAR2]]
; CHECK-NEXT: mov [[VECO]].h[3], [[SCALAR3]]
; CHECK-NEXT: mls v0.4h, [[VECO]].4h, v1.4h
; CHECK: ret

; VBITS_EQ_128-LABEL: urem_v4i16:
; VBITS_EQ_128:         ushll v2.4s, v1.4h, #0
; VBITS_EQ_128-NEXT:    ushll v3.4s, v0.4h, #0
; VBITS_EQ_128-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    udivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    xtn v2.4h, v2.4s
; VBITS_EQ_128-NEXT:    mls v0.4h, v2.4h, v1.4h
; VBITS_EQ_128-NEXT:    ret

  %res = urem <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

define <8 x i16> @urem_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: urem_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; CHECK-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; CHECK-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK-NEXT: mls v0.8h, v2.8h, v1.8h
; CHECK: ret

; VBITS_EQ_128-LABEL: urem_v8i16:
; VBITS_EQ_128:         ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    uunpkhi z2.s, z1.h
; VBITS_EQ_128-NEXT:    uunpkhi z3.s, z0.h
; VBITS_EQ_128-NEXT:    uunpklo z4.s, z1.h
; VBITS_EQ_128-NEXT:    udivr z2.s, p0/m, z2.s, z3.s
; VBITS_EQ_128-NEXT:    uunpklo z5.s, z0.h
; VBITS_EQ_128-NEXT:    movprfx z3, z5
; VBITS_EQ_128-NEXT:    udiv z3.s, p0/m, z3.s, z4.s
; VBITS_EQ_128-NEXT:    uzp1 z2.h, z3.h, z2.h
; VBITS_EQ_128-NEXT:    mls v0.8h, v2.8h, v1.8h
; VBITS_EQ_128-NEXT:    ret

  %res = urem <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @urem_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: urem_v16i16:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_256-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_256-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_256-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_512-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = urem <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @urem_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: urem_v32i16:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_512-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_512-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_512-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_1024-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = urem <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @urem_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: urem_v64i16:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: movprfx [[OP1_LO_:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_1024-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_]].s, [[OP2_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_EQ_1024-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_EQ_1024-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_2048-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO]].s, [[OP1_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: mul [[OP2]].h, [[PG1]]/m, [[OP2]].h, [[UZP1]].h
; VBITS_GE_2048-NEXT: sub [[OP1]].h, [[PG1]]/m, [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: st1h { [[OP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = urem <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @urem_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: urem_v128i16:
; VBITS_EQ_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: udivr   [[RES_HI:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: movprfx [[RES_LO:z[0-9]+]], [[OP1_LO]]
; VBITS_EQ_2048-NEXT: udiv    [[RES_LO:z[0-9]+]].s, [[PG1]]/m, [[RES_LO]].s, [[OP2_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[ZIP:z[0-9]+]].h, [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_2048-NEXT: mul [[MUL:z[0-9]+]].h, [[PG]]/m, [[OP2]].h, [[ZIP]].h
; VBITS_EQ_2048-NEXT: sub [[RES:z[0-9]+]].h, [[PG]]/m, [[OP1]].h, [[MUL]].h
; VBITS_EQ_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = urem <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector v2i32 udiv are not legal for NEON so use SVE when available.
define <2 x i32> @urem_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: urem_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], z0
; CHECK-NEXT: udiv z2.s, [[PG]]/m, [[PFX]].s, z1.s
; CHECK-NEXT: mls v0.2s, v2.2s, v1.2s
; CHECK: ret

; VBITS_EQ_128-LABEL: urem_v2i32:
; VBITS_EQ_128:         ptrue p0.s, vl2
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    udiv z2.s, p0/m, z2.s, z1.s
; VBITS_EQ_128-NEXT:    mls v0.2s, v2.2s, v1.2s
; VBITS_EQ_128-NEXT:    ret

  %res = urem <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Vector v4i32 udiv are not legal for NEON so use SVE when available.
define <4 x i32> @urem_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: urem_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], z0
; CHECK-NEXT: udiv z2.s, [[PG]]/m, [[PFX]].s, z1.s
; CHECK-NEXT: mls v0.4s, v2.4s, v1.4s
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: urem_v4i32:
; VBITS_EQ_128:         ptrue p0.s, vl4
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    udiv z2.s, p0/m, z2.s, z1.s
; VBITS_EQ_128-NEXT:    mls v0.4s, v2.4s, v1.4s
; VBITS_EQ_128-NEXT:    ret

  %res = urem <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @urem_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: urem_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; CHECK-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; CHECK-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = urem <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @urem_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: urem_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_512-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_512-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_512-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = urem <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @urem_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: urem_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_1024-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_1024-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = urem <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @urem_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: urem_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_2048-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[PFX]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: mul [[MUL:z[0-9]+]].s, [[PG]]/m, [[OP2]].s, [[DIV]].s
; VBITS_GE_2048-NEXT: sub [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[MUL]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = urem <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 udiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for the 128 bits case here.
define <1 x i64> @urem_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: urem_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, z1.d
; CHECK-NEXT: mul z1.d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub d0, d0, d1
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: urem_v1i64:
; VBITS_EQ_128:         ptrue p0.d, vl1
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    udiv z2.d, p0/m, z2.d, z1.d
; VBITS_EQ_128-NEXT:    fmov x8, d2
; VBITS_EQ_128-NEXT:    fmov x9, d1
; VBITS_EQ_128-NEXT:    mul x8, x8, x9
; VBITS_EQ_128-NEXT:    fmov d1, x8
; VBITS_EQ_128-NEXT:    sub d0, d0, d1
; VBITS_EQ_128-NEXT:    ret

  %res = urem <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Vector i64 udiv are not legal for NEON so use SVE when available.
; FIXME: We should be able to improve the codegen for the 128 bits case here.
define <2 x i64> @urem_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: urem_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, z1.d
; CHECK-NEXT: mul z1.d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub v0.2d, v0.2d, v1.2d
; CHECK-NEXT: ret

; VBITS_EQ_128-LABEL: urem_v2i64:
; VBITS_EQ_128:         ptrue p0.d, vl2
; VBITS_EQ_128-NEXT:    movprfx z2, z0
; VBITS_EQ_128-NEXT:    udiv z2.d, p0/m, z2.d, z1.d
; VBITS_EQ_128-NEXT:    fmov x9, d2
; VBITS_EQ_128-NEXT:    fmov x10, d1
; VBITS_EQ_128-NEXT:    mov x8, v2.d[1]
; VBITS_EQ_128-NEXT:    mov x11, v1.d[1]
; VBITS_EQ_128-NEXT:    mul x9, x9, x10
; VBITS_EQ_128-NEXT:    mul x8, x8, x11
; VBITS_EQ_128-NEXT:    fmov d1, x9
; VBITS_EQ_128-NEXT:    mov v1.d[1], x8
; VBITS_EQ_128-NEXT:    sub v0.2d, v0.2d, v1.2d
; VBITS_EQ_128-NEXT:    ret

  %res = urem <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @urem_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: urem_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; CHECK-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; CHECK-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; CHECK-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = urem <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @urem_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: urem_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_512-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_512-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_512-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = urem <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @urem_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: urem_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_1024-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_1024-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = urem <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @urem_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: urem_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: movprfx [[PFX:z[0-9]+]], [[OP1]]
; VBITS_GE_2048-NEXT: udiv [[DIV:z[0-9]+]].d, [[PG]]/m, [[PFX]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: mul [[MUL:z[0-9]+]].d, [[PG]]/m, [[OP2]].d, [[DIV]].d
; VBITS_GE_2048-NEXT: sub [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[MUL]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = urem <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
