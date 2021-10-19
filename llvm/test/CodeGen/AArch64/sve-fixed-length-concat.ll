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
; NO_SVE-NOT: ptrue

;
; i8
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @concat_v8i8(<4 x i8> %op1, <4 x i8> %op2) #0 {
; CHECK-LABEL: concat_v8i8:
; CHECK: uzp1 v0.8b, v0.8b, v1.8b
; CHECK-NEXT: ret
  %res = shufflevector <4 x i8> %op1, <4 x i8> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %res
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @concat_v16i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: concat_v16i8:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <8 x i8> %op1, <8 x i8> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                 i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define void @concat_v32i8(<16 x i8>* %a, <16 x i8>* %b, <32 x i8>* %c) #0 {
; CHECK-LABEL: concat_v32i8:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].b, vl16
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].b, [[PG1]], z[[OP1]].b, z[[OP2]].b
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].b, vl32
; CHECK-NEXT: st1b { [[RES]].b }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <16 x i8>, <16 x i8>* %a
  %op2 = load <16 x i8>, <16 x i8>* %b
  %res = shufflevector <16 x i8> %op1, <16 x i8> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                   i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                   i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i8> %res, <32 x i8>* %c
  ret void
}

define void @concat_v64i8(<32 x i8>* %a, <32 x i8>* %b, <64 x i8>* %c) #0 {
; CHECK-LABEL: concat_v64i8:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_GE_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].b, [[PG1]], [[OP1]].b, [[OP2]].b
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = shufflevector <32 x i8> %op1, <32 x i8> %op2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                   i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                   i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                   i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                   i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                   i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                   i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x i8> %res, <64 x i8>* %c
  ret void
}

define void @concat_v128i8(<64 x i8>* %a, <64 x i8>* %b, <128 x i8>* %c) #0 {
; CHECK-LABEL: concat_v128i8:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].b, [[PG1]], [[OP1]].b, [[OP2]].b
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = shufflevector <64 x i8> %op1, <64 x i8> %op2, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                    i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                    i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                    i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                    i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63,
                                                                    i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71,
                                                                    i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79,
                                                                    i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87,
                                                                    i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95,
                                                                    i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103,
                                                                    i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111,
                                                                    i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119,
                                                                    i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  store <128 x i8> %res, <128 x i8>* %c
  ret void
}

define void @concat_v256i8(<128 x i8>* %a, <128 x i8>* %b, <256 x i8>* %c) #0 {
; CHECK-LABEL: concat_v256i8:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].b, [[PG1]], [[OP1]].b, [[OP2]].b
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = shufflevector <128 x i8> %op1, <128 x i8> %op2, <256 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                      i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                      i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                      i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                      i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                      i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                      i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63,
                                                                      i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71,
                                                                      i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79,
                                                                      i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87,
                                                                      i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95,
                                                                      i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103,
                                                                      i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111,
                                                                      i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119,
                                                                      i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127,
                                                                      i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135,
                                                                      i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143,
                                                                      i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151,
                                                                      i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159,
                                                                      i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167,
                                                                      i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175,
                                                                      i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183,
                                                                      i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191,
                                                                      i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199,
                                                                      i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207,
                                                                      i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215,
                                                                      i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223,
                                                                      i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231,
                                                                      i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239,
                                                                      i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247,
                                                                      i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>
  store <256 x i8> %res, <256 x i8>* %c
  ret void
}

;
; i16
;

; Don't use SVE for 64-bit vectors.
define <4 x i16> @concat_v4i16(<2 x i16> %op1, <2 x i16> %op2) #0 {
; CHECK-LABEL: concat_v4i16:
; CHECK: uzp1 v0.4h, v0.4h, v1.4h
; CHECK-NEXT: ret
  %res = shufflevector <2 x i16> %op1, <2 x i16> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @concat_v8i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: concat_v8i16:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <4 x i16> %op1, <4 x i16> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %res
}

define void @concat_v16i16(<8 x i16>* %a, <8 x i16>* %b, <16 x i16>* %c) #0 {
; CHECK-LABEL: concat_v16i16:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].h, vl8
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], z[[OP1]].h, z[[OP2]].h
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %op2 = load <8 x i16>, <8 x i16>* %b
  %res = shufflevector <8 x i16> %op1, <8 x i16> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i16> %res, <16 x i16>* %c
  ret void
}

define void @concat_v32i16(<16 x i16>* %a, <16 x i16>* %b, <32 x i16>* %c) #0 {
; CHECK-LABEL: concat_v32i16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = shufflevector <16 x i16> %op1, <16 x i16> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                     i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                     i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i16> %res, <32 x i16>* %c
  ret void
}

define void @concat_v64i16(<32 x i16>* %a, <32 x i16>* %b, <64 x i16>* %c) #0 {
; CHECK-LABEL: concat_v64i16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = shufflevector <32 x i16> %op1, <32 x i16> %op2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                     i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                     i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                     i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                     i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                     i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                     i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x i16> %res, <64 x i16>* %c
  ret void
}

define void @concat_v128i16(<64 x i16>* %a, <64 x i16>* %b, <128 x i16>* %c) #0 {
; CHECK-LABEL: concat_v128i16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = shufflevector <64 x i16> %op1, <64 x i16> %op2, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                      i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                      i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                      i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                      i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                      i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                      i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63,
                                                                      i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71,
                                                                      i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79,
                                                                      i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87,
                                                                      i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95,
                                                                      i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103,
                                                                      i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111,
                                                                      i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119,
                                                                      i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  store <128 x i16> %res, <128 x i16>* %c
  ret void
}

;
; i32
;

; Don't use SVE for 64-bit vectors.
define <2 x i32> @concat_v2i32(<1 x i32> %op1, <1 x i32> %op2) #0 {
; CHECK-LABEL: concat_v2i32:
; CHECK: zip1 v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = shufflevector <1 x i32> %op1, <1 x i32> %op2, <2 x i32> <i32 0, i32 1>
  ret <2 x i32> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @concat_v4i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: concat_v4i32:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <2 x i32> %op1, <2 x i32> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

define void @concat_v8i32(<4 x i32>* %a, <4 x i32>* %b, <8 x i32>* %c) #0 {
; CHECK-LABEL: concat_v8i32:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], z[[OP1]].s, z[[OP2]].s
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %op2 = load <4 x i32>, <4 x i32>* %b
  %res = shufflevector <4 x i32> %op1, <4 x i32> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %res, <8 x i32>* %c
  ret void
}

define void @concat_v16i32(<8 x i32>* %a, <8 x i32>* %b, <16 x i32>* %c) #0 {
; CHECK-LABEL: concat_v16i32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = shufflevector <8 x i32> %op1, <8 x i32> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i32> %res, <16 x i32>* %c
  ret void
}

define void @concat_v32i32(<16 x i32>* %a, <16 x i32>* %b, <32 x i32>* %c) #0 {
; CHECK-LABEL: concat_v32i32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = shufflevector <16 x i32> %op1, <16 x i32> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                     i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                     i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %res, <32 x i32>* %c
  ret void
}

define void @concat_v64i32(<32 x i32>* %a, <32 x i32>* %b, <64 x i32>* %c) #0 {
; CHECK-LABEL: concat_v64i32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = shufflevector <32 x i32> %op1, <32 x i32> %op2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                     i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                     i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                     i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                     i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                     i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                     i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x i32> %res, <64 x i32>* %c
  ret void
}

;
; i64
;

; Don't use SVE for 128-bit vectors.
define <2 x i64> @concat_v2i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: concat_v2i64:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <1 x i64> %op1, <1 x i64> %op2, <2 x i32> <i32 0, i32 1>
  ret <2 x i64> %res
}

define void @concat_v4i64(<2 x i64>* %a, <2 x i64>* %b, <4 x i64>* %c) #0 {
; CHECK-LABEL: concat_v4i64:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].d, vl2
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], z[[OP1]].d, z[[OP2]].d
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <2 x i64>, <2 x i64>* %a
  %op2 = load <2 x i64>, <2 x i64>* %b
  %res = shufflevector <2 x i64> %op1, <2 x i64> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %res, <4 x i64>* %c
  ret void
}

define void @concat_v8i64(<4 x i64>* %a, <4 x i64>* %b, <8 x i64>* %c) #0 {
; CHECK-LABEL: concat_v8i64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = shufflevector <4 x i64> %op1, <4 x i64> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i64> %res, <8 x i64>* %c
  ret void
}

define void @concat_v16i64(<8 x i64>* %a, <8 x i64>* %b, <16 x i64>* %c) #0 {
; CHECK-LABEL: concat_v16i64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = shufflevector <8 x i64> %op1, <8 x i64> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i64> %res, <16 x i64>* %c
  ret void
}

define void @concat_v32i64(<16 x i64>* %a, <16 x i64>* %b, <32 x i64>* %c) #0 {
; CHECK-LABEL: concat_v32i64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = shufflevector <16 x i64> %op1, <16 x i64> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                     i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                     i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i64> %res, <32 x i64>* %c
  ret void
}

;
; f16
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @concat_v4f16(<2 x half> %op1, <2 x half> %op2) #0 {
; CHECK-LABEL: concat_v4f16:
; CHECK: zip1 v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = shufflevector <2 x half> %op1, <2 x half> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x half> %res
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @concat_v8f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: concat_v8f16:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <4 x half> %op1, <4 x half> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x half> %res
}

define void @concat_v16f16(<8 x half>* %a, <8 x half>* %b, <16 x half>* %c) #0 {
; CHECK-LABEL: concat_v16f16:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].h, vl8
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], z[[OP1]].h, z[[OP2]].h
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <8 x half>, <8 x half>* %a
  %op2 = load <8 x half>, <8 x half>* %b
  %res = shufflevector <8 x half> %op1, <8 x half> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x half> %res, <16 x half>* %c
  ret void
}

define void @concat_v32f16(<16 x half>* %a, <16 x half>* %b, <32 x half>* %c) #0 {
; CHECK-LABEL: concat_v32f16:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %res = shufflevector <16 x half> %op1, <16 x half> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                       i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                       i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x half> %res, <32 x half>* %c
  ret void
}

define void @concat_v64f16(<32 x half>* %a, <32 x half>* %b, <64 x half>* %c) #0 {
; CHECK-LABEL: concat_v64f16:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %res = shufflevector <32 x half> %op1, <32 x half> %op2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                       i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                       i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                       i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                       i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                       i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                       i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x half> %res, <64 x half>* %c
  ret void
}

define void @concat_v128f16(<64 x half>* %a, <64 x half>* %b, <128 x half>* %c) #0 {
; CHECK-LABEL: concat_v128f16:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].h, [[PG1]], [[OP1]].h, [[OP2]].h
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %res = shufflevector <64 x half> %op1, <64 x half> %op2, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                        i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                        i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                        i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                        i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                        i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                        i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                        i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63,
                                                                        i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71,
                                                                        i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79,
                                                                        i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87,
                                                                        i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95,
                                                                        i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103,
                                                                        i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111,
                                                                        i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119,
                                                                        i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
  store <128 x half> %res, <128 x half>* %c
  ret void
}

;
; i32
;

; Don't use SVE for 64-bit vectors.
define <2 x float> @concat_v2f32(<1 x float> %op1, <1 x float> %op2) #0 {
; CHECK-LABEL: concat_v2f32:
; CHECK: zip1 v0.2s, v0.2s, v1.2s
; CHECK-NEXT: ret
  %res = shufflevector <1 x float> %op1, <1 x float> %op2, <2 x i32> <i32 0, i32 1>
  ret <2 x float> %res
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @concat_v4f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: concat_v4f32:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <2 x float> %op1, <2 x float> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %res
}

define void @concat_v8f32(<4 x float>* %a, <4 x float>* %b, <8 x float>* %c) #0 {
; CHECK-LABEL: concat_v8f32:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], z[[OP1]].s, z[[OP2]].s
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <4 x float>, <4 x float>* %a
  %op2 = load <4 x float>, <4 x float>* %b
  %res = shufflevector <4 x float> %op1, <4 x float> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x float> %res, <8 x float>* %c
  ret void
}

define void @concat_v16f32(<8 x float>* %a, <8 x float>* %b, <16 x float>* %c) #0 {
; CHECK-LABEL: concat_v16f32:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].s, vl8
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %res = shufflevector <8 x float> %op1, <8 x float> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x float> %res, <16 x float>* %c
  ret void
}

define void @concat_v32f32(<16 x float>* %a, <16 x float>* %b, <32 x float>* %c) #0 {
; CHECK-LABEL: concat_v32f32:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].s, vl16
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %res = shufflevector <16 x float> %op1, <16 x float> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                         i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                         i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                         i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x float> %res, <32 x float>* %c
  ret void
}

define void @concat_v64f32(<32 x float>* %a, <32 x float>* %b, <64 x float>* %c) #0 {
; CHECK-LABEL: concat_v64f32:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].s, vl32
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].s, [[PG1]], [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %res = shufflevector <32 x float> %op1, <32 x float> %op2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                         i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                         i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                         i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31,
                                                                         i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39,
                                                                         i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47,
                                                                         i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55,
                                                                         i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <64 x float> %res, <64 x float>* %c
  ret void
}

;
; f64
;

; Don't use SVE for 128-bit vectors.
define <2 x double> @concat_v2f64(<1 x double> %op1, <1 x double> %op2) #0 {
; CHECK-LABEL: concat_v2f64:
; CHECK: mov v0.d[1], v1.d[0]
; CHECK-NEXT: ret
  %res = shufflevector <1 x double> %op1, <1 x double> %op2, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %res
}

define void @concat_v4f64(<2 x double>* %a, <2 x double>* %b, <4 x double>* %c) #0 {
; CHECK-LABEL: concat_v4f64:
; CHECK: ldr q[[OP2:[0-9]+]], [x1]
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].d, vl2
; CHECK-NEXT: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], z[[OP1]].d, z[[OP2]].d
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; CHECK-NEXT: ret
  %op1 = load <2 x double>, <2 x double>* %a
  %op2 = load <2 x double>, <2 x double>* %b
  %res = shufflevector <2 x double> %op1, <2 x double> %op2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x double> %res, <4 x double>* %c
  ret void
}

define void @concat_v8f64(<4 x double>* %a, <4 x double>* %b, <8 x double>* %c) #0 {
; CHECK-LABEL: concat_v8f64:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].d, vl4
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_512-NEXT: ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %res = shufflevector <4 x double> %op1, <4 x double> %op2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x double> %res, <8 x double>* %c
  ret void
}

define void @concat_v16f64(<8 x double>* %a, <8 x double>* %b, <16 x double>* %c) #0 {
; CHECK-LABEL: concat_v16f64:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].d, vl8
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %res = shufflevector <8 x double> %op1, <8 x double> %op2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                         i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x double> %res, <16 x double>* %c
  ret void
}

define void @concat_v32f64(<16 x double>* %a, <16 x double>* %b, <32 x double>* %c) #0 {
; CHECK-LABEL: concat_v32f64:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].d, vl16
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: splice [[RES:z[0-9]+]].d, [[PG1]], [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG2]], [x2]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %res = shufflevector <16 x double> %op1, <16 x double> %op2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                           i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                           i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                           i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x double> %res, <32 x double>* %c
  ret void
}

;
; undef
;

define void @concat_v32i8_undef(<16 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: concat_v32i8_undef:
; CHECK: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: st1b { z[[OP1]].b }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <16 x i8>, <16 x i8>* %a
  %res = shufflevector <16 x i8> %op1, <16 x i8> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                    i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                    i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i8> %res, <32 x i8>* %b
  ret void
}

define void @concat_v16i16_undef(<8 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: concat_v16i16_undef:
; CHECK: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { z[[OP1]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i16>, <8 x i16>* %a
  %res = shufflevector <8 x i16> %op1, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                    i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @concat_v8i32_undef(<4 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: concat_v8i32_undef:
; CHECK: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { z[[OP1]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i32>, <4 x i32>* %a
  %res = shufflevector <4 x i32> %op1, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @concat_v4i64_undef(<2 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: concat_v4i64_undef:
; CHECK: ldr q[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { z[[OP1]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <2 x i64>, <2 x i64>* %a
  %res = shufflevector <2 x i64> %op1, <2 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

;
; > 2 operands
;

define void @concat_v32i8_4op(<8 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: concat_v32i8_4op:
; CHECK: ldr d[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: st1b { z[[OP1]].b }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <8 x i8>, <8 x i8>* %a
  %shuffle = shufflevector <8 x i8> %op1, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %res = shufflevector <16 x i8> %shuffle, <16 x i8> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                        i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                                                        i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                                                        i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i8> %res, <32 x i8>* %b
  ret void
}

define void @concat_v16i16_4op(<4 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: concat_v16i16_4op:
; CHECK: ldr d[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { z[[OP1]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <4 x i16>, <4 x i16>* %a
  %shuffle = shufflevector <4 x i16> %op1, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %res = shufflevector <8 x i16> %shuffle, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                                                        i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <16 x i16> %res, <16 x i16>* %b
  ret void
}

define void @concat_v8i32_4op(<2 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: concat_v8i32_4op:
; CHECK: ldr d[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { z[[OP1]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <2 x i32>, <2 x i32>* %a
  %shuffle = shufflevector <2 x i32> %op1, <2 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %res = shufflevector <4 x i32> %shuffle, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i32> %res, <8 x i32>* %b
  ret void
}

define void @concat_v4i64_4op(<1 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: concat_v4i64_4op:
; CHECK: ldr d[[OP1:[0-9]+]], [x0]
; CHECK-NEXT: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { z[[OP1]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %op1 = load <1 x i64>, <1 x i64>* %a
  %shuffle = shufflevector <1 x i64> %op1, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %res = shufflevector <2 x i64> %shuffle, <2 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i64> %res, <4 x i64>* %b
  ret void
}

attributes #0 = { "target-features"="+sve" }
