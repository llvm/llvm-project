; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; DUP (integer)
;

; Don't use SVE for 64-bit vectors.
define <8 x i8> @splat_v8i8(i8 %a) #0 {
; CHECK-LABEL: splat_v8i8:
; CHECK: dup v0.8b, w0
; CHECK-NEXT: ret
  %insert = insertelement <8 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <8 x i8> %insert, <8 x i8> undef, <8 x i32> zeroinitializer
  ret <8 x i8> %splat
}

; Don't use SVE for 128-bit vectors.
define <16 x i8> @splat_v16i8(i8 %a) #0 {
; CHECK-LABEL: splat_v16i8:
; CHECK: dup v0.16b, w0
; CHECK-NEXT: ret
  %insert = insertelement <16 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <16 x i8> %insert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat
}

define void @splat_v32i8(i8 %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: splat_v32i8:
; CHECK-DAG: mov [[RES:z[0-9]+]].b, w0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; CHECK-NEXT: st1b { [[RES]].b }, [[PG]], [x1]
; CHECK-NEXT: ret
  %insert = insertelement <32 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <32 x i8> %insert, <32 x i8> undef, <32 x i32> zeroinitializer
  store <32 x i8> %splat, <32 x i8>* %b
  ret void
}

define void @splat_v64i8(i8 %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: splat_v64i8:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].b, w0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].b, w0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].b, vl32
; VBITS_EQ_256-DAG: mov w[[OFFSET_HI:[0-9]+]], #32
; VBITS_EQ_256-DAG: st1b { [[RES]].b }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1b { [[RES]].b }, [[PG]], [x1, x[[OFFSET_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <64 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <64 x i8> %insert, <64 x i8> undef, <64 x i32> zeroinitializer
  store <64 x i8> %splat, <64 x i8>* %b
  ret void
}

define void @splat_v128i8(i8 %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: splat_v128i8:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].b, w0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].b, vl128
; VBITS_GE_1024-NEXT: st1b { [[RES]].b }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <128 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <128 x i8> %insert, <128 x i8> undef, <128 x i32> zeroinitializer
  store <128 x i8> %splat, <128 x i8>* %b
  ret void
}

define void @splat_v256i8(i8 %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: splat_v256i8:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].b, w0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].b, vl256
; VBITS_GE_2048-NEXT: st1b { [[RES]].b }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <256 x i8> undef, i8 %a, i64 0
  %splat = shufflevector <256 x i8> %insert, <256 x i8> undef, <256 x i32> zeroinitializer
  store <256 x i8> %splat, <256 x i8>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define <4 x i16> @splat_v4i16(i16 %a) #0 {
; CHECK-LABEL: splat_v4i16:
; CHECK: dup v0.4h, w0
; CHECK-NEXT: ret
  %insert = insertelement <4 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <4 x i16> %insert, <4 x i16> undef, <4 x i32> zeroinitializer
  ret <4 x i16> %splat
}

; Don't use SVE for 128-bit vectors.
define <8 x i16> @splat_v8i16(i16 %a) #0 {
; CHECK-LABEL: splat_v8i16:
; CHECK: dup v0.8h, w0
; CHECK-NEXT: ret
  %insert = insertelement <8 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <8 x i16> %insert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat
}

define void @splat_v16i16(i16 %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: splat_v16i16:
; CHECK-DAG: mov [[RES:z[0-9]+]].h, w0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; CHECK-NEXT: ret
  %insert = insertelement <16 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <16 x i16> %insert, <16 x i16> undef, <16 x i32> zeroinitializer
  store <16 x i16> %splat, <16 x i16>* %b
  ret void
}

define void @splat_v32i16(i16 %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: splat_v32i16:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].h, w0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].h, w0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1h { [[RES]].h }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <32 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <32 x i16> %insert, <32 x i16> undef, <32 x i32> zeroinitializer
  store <32 x i16> %splat, <32 x i16>* %b
  ret void
}

define void @splat_v64i16(i16 %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: splat_v64i16:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].h, w0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <64 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <64 x i16> %insert, <64 x i16> undef, <64 x i32> zeroinitializer
  store <64 x i16> %splat, <64 x i16>* %b
  ret void
}

define void @splat_v128i16(i16 %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: splat_v128i16:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].h, w0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <128 x i16> undef, i16 %a, i64 0
  %splat = shufflevector <128 x i16> %insert, <128 x i16> undef, <128 x i32> zeroinitializer
  store <128 x i16> %splat, <128 x i16>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x i32> @splat_v2i32(i32 %a) #0 {
; CHECK-LABEL: splat_v2i32:
; CHECK: dup v0.2s, w0
; CHECK-NEXT: ret
  %insert = insertelement <2 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <2 x i32> %insert, <2 x i32> undef, <2 x i32> zeroinitializer
  ret <2 x i32> %splat
}

; Don't use SVE for 128-bit vectors.
define <4 x i32> @splat_v4i32(i32 %a) #0 {
; CHECK-LABEL: splat_v4i32:
; CHECK: dup v0.4s, w0
; CHECK-NEXT: ret
  %insert = insertelement <4 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat
}

define void @splat_v8i32(i32 %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: splat_v8i32:
; CHECK-DAG: mov [[RES:z[0-9]+]].s, w0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; CHECK-NEXT: ret
  %insert = insertelement <8 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <8 x i32> %insert, <8 x i32> undef, <8 x i32> zeroinitializer
  store <8 x i32> %splat, <8 x i32>* %b
  ret void
}

define void @splat_v16i32(i32 %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: splat_v16i32:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].s, w0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].s, w0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1w { [[RES]].s }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <16 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <16 x i32> %insert, <16 x i32> undef, <16 x i32> zeroinitializer
  store <16 x i32> %splat, <16 x i32>* %b
  ret void
}

define void @splat_v32i32(i32 %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: splat_v32i32:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].s, w0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <32 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <32 x i32> %insert, <32 x i32> undef, <32 x i32> zeroinitializer
  store <32 x i32> %splat, <32 x i32>* %b
  ret void
}

define void @splat_v64i32(i32 %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: splat_v64i32:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].s, w0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <64 x i32> undef, i32 %a, i64 0
  %splat = shufflevector <64 x i32> %insert, <64 x i32> undef, <64 x i32> zeroinitializer
  store <64 x i32> %splat, <64 x i32>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x i64> @splat_v1i64(i64 %a) #0 {
; CHECK-LABEL: splat_v1i64:
; CHECK: fmov d0, x0
; CHECK-NEXT: ret
  %insert = insertelement <1 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <1 x i64> %insert, <1 x i64> undef, <1 x i32> zeroinitializer
  ret <1 x i64> %splat
}

; Don't use SVE for 128-bit vectors.
define <2 x i64> @splat_v2i64(i64 %a) #0 {
; CHECK-LABEL: splat_v2i64:
; CHECK: dup v0.2d, x0
; CHECK-NEXT: ret
  %insert = insertelement <2 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <2 x i64> %insert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat
}

define void @splat_v4i64(i64 %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: splat_v4i64:
; CHECK-DAG: mov [[RES:z[0-9]+]].d, x0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; CHECK-NEXT: ret
  %insert = insertelement <4 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <4 x i64> %insert, <4 x i64> undef, <4 x i32> zeroinitializer
  store <4 x i64> %splat, <4 x i64>* %b
  ret void
}

define void @splat_v8i64(i64 %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: splat_v8i64:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].d, x0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].d, x0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x1, #32
; VBITS_EQ_256-DAG: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_EQ_256-DAG: st1d { [[RES]].d }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <8 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <8 x i64> %insert, <8 x i64> undef, <8 x i32> zeroinitializer
  store <8 x i64> %splat, <8 x i64>* %b
  ret void
}

define void @splat_v16i64(i64 %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: splat_v16i64:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].d, x0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <16 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <16 x i64> %insert, <16 x i64> undef, <16 x i32> zeroinitializer
  store <16 x i64> %splat, <16 x i64>* %b
  ret void
}

define void @splat_v32i64(i64 %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: splat_v32i64:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].d, x0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x1]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <32 x i64> undef, i64 %a, i64 0
  %splat = shufflevector <32 x i64> %insert, <32 x i64> undef, <32 x i32> zeroinitializer
  store <32 x i64> %splat, <32 x i64>* %b
  ret void
}

;
; DUP (floating-point)
;

; Don't use SVE for 64-bit vectors.
define <4 x half> @splat_v4f16(half %a) #0 {
; CHECK-LABEL: splat_v4f16:
; CHECK: dup v0.4h, v0.h[0]
; CHECK-NEXT: ret
  %insert = insertelement <4 x half> undef, half %a, i64 0
  %splat = shufflevector <4 x half> %insert, <4 x half> undef, <4 x i32> zeroinitializer
  ret <4 x half> %splat
}

; Don't use SVE for 128-bit vectors.
define <8 x half> @splat_v8f16(half %a) #0 {
; CHECK-LABEL: splat_v8f16:
; CHECK: dup v0.8h, v0.h[0]
; CHECK-NEXT: ret
  %insert = insertelement <8 x half> undef, half %a, i64 0
  %splat = shufflevector <8 x half> %insert, <8 x half> undef, <8 x i32> zeroinitializer
  ret <8 x half> %splat
}

define void @splat_v16f16(half %a, <16 x half>* %b) #0 {
; CHECK-LABEL: splat_v16f16:
; CHECK-DAG: mov [[RES:z[0-9]+]].h, h0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; CHECK-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; CHECK-NEXT: ret
  %insert = insertelement <16 x half> undef, half %a, i64 0
  %splat = shufflevector <16 x half> %insert, <16 x half> undef, <16 x i32> zeroinitializer
  store <16 x half> %splat, <16 x half>* %b
  ret void
}

define void @splat_v32f16(half %a, <32 x half>* %b) #0 {
; CHECK-LABEL: splat_v32f16:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].h, h0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].h, h0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].h, vl16
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1h { [[RES]].h }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <32 x half> undef, half %a, i64 0
  %splat = shufflevector <32 x half> %insert, <32 x half> undef, <32 x i32> zeroinitializer
  store <32 x half> %splat, <32 x half>* %b
  ret void
}

define void @splat_v64f16(half %a, <64 x half>* %b) #0 {
; CHECK-LABEL: splat_v64f16:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].h, h0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].h, vl64
; VBITS_GE_1024-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <64 x half> undef, half %a, i64 0
  %splat = shufflevector <64 x half> %insert, <64 x half> undef, <64 x i32> zeroinitializer
  store <64 x half> %splat, <64 x half>* %b
  ret void
}

define void @splat_v128f16(half %a, <128 x half>* %b) #0 {
; CHECK-LABEL: splat_v128f16:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].h, h0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_GE_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <128 x half> undef, half %a, i64 0
  %splat = shufflevector <128 x half> %insert, <128 x half> undef, <128 x i32> zeroinitializer
  store <128 x half> %splat, <128 x half>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define <2 x float> @splat_v2f32(float %a, <2 x float> %op2) #0 {
; CHECK-LABEL: splat_v2f32:
; CHECK: dup v0.2s, v0.s[0]
; CHECK-NEXT: ret
  %insert = insertelement <2 x float> undef, float %a, i64 0
  %splat = shufflevector <2 x float> %insert, <2 x float> undef, <2 x i32> zeroinitializer
  ret <2 x float> %splat
}

; Don't use SVE for 128-bit vectors.
define <4 x float> @splat_v4f32(float %a, <4 x float> %op2) #0 {
; CHECK-LABEL: splat_v4f32:
; CHECK: dup v0.4s, v0.s[0]
; CHECK-NEXT: ret
  %insert = insertelement <4 x float> undef, float %a, i64 0
  %splat = shufflevector <4 x float> %insert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat
}

define void @splat_v8f32(float %a, <8 x float>* %b) #0 {
; CHECK-LABEL: splat_v8f32:
; CHECK-DAG: mov [[RES:z[0-9]+]].s, s0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %insert = insertelement <8 x float> undef, float %a, i64 0
  %splat = shufflevector <8 x float> %insert, <8 x float> undef, <8 x i32> zeroinitializer
  store <8 x float> %splat, <8 x float>* %b
  ret void
}

define void @splat_v16f32(float %a, <16 x float>* %b) #0 {
; CHECK-LABEL: splat_v16f32:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].s, s0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].s, s0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1w { [[RES]].s }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <16 x float> undef, float %a, i64 0
  %splat = shufflevector <16 x float> %insert, <16 x float> undef, <16 x i32> zeroinitializer
  store <16 x float> %splat, <16 x float>* %b
  ret void
}

define void @splat_v32f32(float %a, <32 x float>* %b) #0 {
; CHECK-LABEL: splat_v32f32:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].s, s0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <32 x float> undef, float %a, i64 0
  %splat = shufflevector <32 x float> %insert, <32 x float> undef, <32 x i32> zeroinitializer
  store <32 x float> %splat, <32 x float>* %b
  ret void
}

define void @splat_v64f32(float %a, <64 x float>* %b) #0 {
; CHECK-LABEL: splat_v64f32:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].s, s0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <64 x float> undef, float %a, i64 0
  %splat = shufflevector <64 x float> %insert, <64 x float> undef, <64 x i32> zeroinitializer
  store <64 x float> %splat, <64 x float>* %b
  ret void
}

; Don't use SVE for 64-bit vectors.
define <1 x double> @splat_v1f64(double %a, <1 x double> %op2) #0 {
; CHECK-LABEL: splat_v1f64:
; CHECK: // %bb.0:
; CHECK-NEXT: ret
  %insert = insertelement <1 x double> undef, double %a, i64 0
  %splat = shufflevector <1 x double> %insert, <1 x double> undef, <1 x i32> zeroinitializer
  ret <1 x double> %splat
}

; Don't use SVE for 128-bit vectors.
define <2 x double> @splat_v2f64(double %a, <2 x double> %op2) #0 {
; CHECK-LABEL: splat_v2f64:
; CHECK: dup v0.2d, v0.d[0]
; CHECK-NEXT: ret
  %insert = insertelement <2 x double> undef, double %a, i64 0
  %splat = shufflevector <2 x double> %insert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat
}

define void @splat_v4f64(double %a, <4 x double>* %b) #0 {
; CHECK-LABEL: splat_v4f64:
; CHECK-DAG: mov [[RES:z[0-9]+]].d, d0
; CHECK-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %insert = insertelement <4 x double> undef, double %a, i64 0
  %splat = shufflevector <4 x double> %insert, <4 x double> undef, <4 x i32> zeroinitializer
  store <4 x double> %splat, <4 x double>* %b
  ret void
}

define void @splat_v8f64(double %a, <8 x double>* %b) #0 {
; CHECK-LABEL: splat_v8f64:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].d, d0
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret

; Ensure sensible type legalisation.
; VBITS_EQ_256-DAG: mov [[RES:z[0-9]+]].d, d0
; VBITS_EQ_256-DAG: ptrue [[PG:p[0-9]+]].d, vl4
; VBITS_EQ_256-DAG: add x[[B_HI:[0-9]+]], x0, #32
; VBITS_EQ_256-DAG: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_EQ_256-DAG: st1d { [[RES]].d }, [[PG]], [x[[B_HI]]
; VBITS_EQ_256-NEXT: ret
  %insert = insertelement <8 x double> undef, double %a, i64 0
  %splat = shufflevector <8 x double> %insert, <8 x double> undef, <8 x i32> zeroinitializer
  store <8 x double> %splat, <8 x double>* %b
  ret void
}

define void @splat_v16f64(double %a, <16 x double>* %b) #0 {
; CHECK-LABEL: splat_v16f64:
; VBITS_GE_1024-DAG: mov [[RES:z[0-9]+]].d, d0
; VBITS_GE_1024-DAG: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %insert = insertelement <16 x double> undef, double %a, i64 0
  %splat = shufflevector <16 x double> %insert, <16 x double> undef, <16 x i32> zeroinitializer
  store <16 x double> %splat, <16 x double>* %b
  ret void
}

define void @splat_v32f64(double %a, <32 x double>* %b) #0 {
; CHECK-LABEL: splat_v32f64:
; VBITS_GE_2048-DAG: mov [[RES:z[0-9]+]].d, d0
; VBITS_GE_2048-DAG: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %insert = insertelement <32 x double> undef, double %a, i64 0
  %splat = shufflevector <32 x double> %insert, <32 x double> undef, <32 x i32> zeroinitializer
  store <32 x double> %splat, <32 x double>* %b
  ret void
}

;
; DUP (integer immediate)
;

define void @splat_imm_v64i8(<64 x i8>* %a) #0 {
; CHECK-LABEL: splat_imm_v64i8:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].b, #1
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].b, vl64
; VBITS_GE_512-NEXT: st1b { [[RES]].b }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <64 x i8> undef, i8 1, i64 0
  %splat = shufflevector <64 x i8> %insert, <64 x i8> undef, <64 x i32> zeroinitializer
  store <64 x i8> %splat, <64 x i8>* %a
  ret void
}

define void @splat_imm_v32i16(<32 x i16>* %a) #0 {
; CHECK-LABEL: splat_imm_v32i16:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].h, #2
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <32 x i16> undef, i16 2, i64 0
  %splat = shufflevector <32 x i16> %insert, <32 x i16> undef, <32 x i32> zeroinitializer
  store <32 x i16> %splat, <32 x i16>* %a
  ret void
}

define void @splat_imm_v16i32(<16 x i32>* %a) #0 {
; CHECK-LABEL: splat_imm_v16i32:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].s, #3
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <16 x i32> undef, i32 3, i64 0
  %splat = shufflevector <16 x i32> %insert, <16 x i32> undef, <16 x i32> zeroinitializer
  store <16 x i32> %splat, <16 x i32>* %a
  ret void
}

define void @splat_imm_v8i64(<8 x i64>* %a) #0 {
; CHECK-LABEL: splat_imm_v8i64:
; VBITS_GE_512-DAG: mov [[RES:z[0-9]+]].d, #4
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <8 x i64> undef, i64 4, i64 0
  %splat = shufflevector <8 x i64> %insert, <8 x i64> undef, <8 x i32> zeroinitializer
  store <8 x i64> %splat, <8 x i64>* %a
  ret void
}

;
; DUP (floating-point immediate)
;

define void @splat_imm_v32f16(<32 x half>* %a) #0 {
; CHECK-LABEL: splat_imm_v32f16:
; VBITS_GE_512-DAG: fmov [[RES:z[0-9]+]].h, #5.00000000
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <32 x half> undef, half 5.0, i64 0
  %splat = shufflevector <32 x half> %insert, <32 x half> undef, <32 x i32> zeroinitializer
  store <32 x half> %splat, <32 x half>* %a
  ret void
}

define void @splat_imm_v16f32(<16 x float>* %a) #0 {
; CHECK-LABEL: splat_imm_v16f32:
; VBITS_GE_512-DAG: fmov [[RES:z[0-9]+]].s, #6.00000000
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <16 x float> undef, float 6.0, i64 0
  %splat = shufflevector <16 x float> %insert, <16 x float> undef, <16 x i32> zeroinitializer
  store <16 x float> %splat, <16 x float>* %a
  ret void
}

define void @splat_imm_v8f64(<8 x double>* %a) #0 {
; CHECK-LABEL: splat_imm_v8f64:
; VBITS_GE_512-DAG: fmov [[RES:z[0-9]+]].d, #7.00000000
; VBITS_GE_512-DAG: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %insert = insertelement <8 x double> undef, double 7.0, i64 0
  %splat = shufflevector <8 x double> %insert, <8 x double> undef, <8 x i32> zeroinitializer
  store <8 x double> %splat, <8 x double>* %a
  ret void
}
attributes #0 = { "target-features"="+sve" }
