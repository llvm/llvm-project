; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_1024,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_2048,VBITS_GE_1024,VBITS_GE_512

target triple = "aarch64-unknown-linux-gnu"

;
; LD1B
;

define void @masked_gather_v2i8(<2 x i8>* %a, <2 x i8*>* %b) #0 {
; CHECK-LABEL: masked_gather_v2i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x1]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    ptrue p0.s, vl2
; CHECK-NEXT:    xtn v0.2s, v0.2d
; CHECK-NEXT:    st1b { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <2 x i8*>, <2 x i8*>* %b
  %vals = call <2 x i8> @llvm.masked.gather.v2i8(<2 x i8*> %ptrs, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i8> undef)
  store <2 x i8> %vals, <2 x i8>* %a
  ret void
}

define void @masked_gather_v4i8(<4 x i8>* %a, <4 x i8*>* %b) #0 {
; CHECK-LABEL: masked_gather_v4i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1]
; CHECK-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    ptrue p0.h, vl4
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    st1b { z0.h }, p0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <4 x i8*>, <4 x i8*>* %b
  %vals = call <4 x i8> @llvm.masked.gather.v4i8(<4 x i8*> %ptrs, i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i8> undef)
  store <4 x i8> %vals, <4 x i8>* %a
  ret void
}

define void @masked_gather_v8i8(<8 x i8>* %a, <8 x i8*>* %b) #0 {
; VBITS_EQ_256-LABEL: masked_gather_v8i8:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; VBITS_EQ_256-NEXT:    ld1b { z1.d }, p0/z, [z1.d]
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z1.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    uzp1 v0.8b, v1.8b, v0.8b
; VBITS_EQ_256-NEXT:    str d0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_gather_v8i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    uzp1 z0.b, z0.b, z0.b
; VBITS_GE_512-NEXT:    str d0, [x0]
; VBITS_GE_512-NEXT:    ret
  %ptrs = load <8 x i8*>, <8 x i8*>* %b
  %vals = call <8 x i8> @llvm.masked.gather.v8i8(<8 x i8*> %ptrs, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i8> undef)
  store <8 x i8> %vals, <8 x i8>* %a
  ret void
}

define void @masked_gather_v16i8(<16 x i8>* %a, <16 x i8*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_gather_v16i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    uzp1 z0.b, z0.b, z0.b
; VBITS_GE_1024-NEXT:    str q0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %ptrs = load <16 x i8*>, <16 x i8*>* %b
  %vals = call <16 x i8> @llvm.masked.gather.v16i8(<16 x i8*> %ptrs, i32 8, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                       i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x i8> undef)
  store <16 x i8> %vals, <16 x i8>* %a
  ret void
}

define void @masked_gather_v32i8(<32 x i8>* %a, <32 x i8*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_gather_v32i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    ld1b { z0.d }, p0/z, [z0.d]
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    uzp1 z0.b, z0.b, z0.b
; VBITS_GE_2048-NEXT:    st1b { z0.b }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %ptrs = load <32 x i8*>, <32 x i8*>* %b
  %vals = call <32 x i8> @llvm.masked.gather.v32i8(<32 x i8*> %ptrs, i32 8, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                       i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                       i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                       i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i8> undef)
  store <32 x i8> %vals, <32 x i8>* %a
  ret void
}

;
; LD1H
;

define void @masked_gather_v2i16(<2 x i16>* %a, <2 x i16*>* %b) #0 {
; CHECK-LABEL: masked_gather_v2i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x1]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    ptrue p0.s, vl2
; CHECK-NEXT:    xtn v0.2s, v0.2d
; CHECK-NEXT:    st1h { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <2 x i16*>, <2 x i16*>* %b
  %vals = call <2 x i16> @llvm.masked.gather.v2i16(<2 x i16*> %ptrs, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i16> undef)
  store <2 x i16> %vals, <2 x i16>* %a
  ret void
}

define void @masked_gather_v4i16(<4 x i16>* %a, <4 x i16*>* %b) #0 {
; CHECK-LABEL: masked_gather_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1]
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    uzp1 z0.h, z0.h, z0.h
; CHECK-NEXT:    str d0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <4 x i16*>, <4 x i16*>* %b
  %vals = call <4 x i16> @llvm.masked.gather.v4i16(<4 x i16*> %ptrs, i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i16> undef)
  store <4 x i16> %vals, <4 x i16>* %a
  ret void
}

define void @masked_gather_v8i16(<8 x i16>* %a, <8 x i16*>* %b) #0 {
; VBITS_EQ_256-LABEL: masked_gather_v8i16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; VBITS_EQ_256-NEXT:    ld1h { z1.d }, p0/z, [z1.d]
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_EQ_256-NEXT:    uzp1 z1.h, z1.h, z1.h
; VBITS_EQ_256-NEXT:    mov v1.d[1], v0.d[0]
; VBITS_EQ_256-NEXT:    str q1, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_gather_v8i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_512-NEXT:    str q0, [x0]
; VBITS_GE_512-NEXT:    ret
  %ptrs = load <8 x i16*>, <8 x i16*>* %b
  %vals = call <8 x i16> @llvm.masked.gather.v8i16(<8 x i16*> %ptrs, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i16> undef)
  store <8 x i16> %vals, <8 x i16>* %a
  ret void
}

define void @masked_gather_v16i16(<16 x i16>* %a, <16 x i16*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_gather_v16i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_1024-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %ptrs = load <16 x i16*>, <16 x i16*>* %b
  %vals = call <16 x i16> @llvm.masked.gather.v16i16(<16 x i16*> %ptrs, i32 8, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x i16> undef)
  store <16 x i16> %vals, <16 x i16>* %a
  ret void
}

define void @masked_gather_v32i16(<32 x i16>* %a, <32 x i16*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_gather_v32i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    ld1h { z0.d }, p0/z, [z0.d]
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    uzp1 z0.h, z0.h, z0.h
; VBITS_GE_2048-NEXT:    st1h { z0.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %ptrs = load <32 x i16*>, <32 x i16*>* %b
  %vals = call <32 x i16> @llvm.masked.gather.v32i16(<32 x i16*> %ptrs, i32 8, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i16> undef)
  store <32 x i16> %vals, <32 x i16>* %a
  ret void
}

;
; LD1W
;

define void @masked_gather_v2i32(<2 x i32>* %a, <2 x i32*>* %b) #0 {
; CHECK-LABEL: masked_gather_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x1]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    xtn v0.2s, v0.2d
; CHECK-NEXT:    str d0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <2 x i32*>, <2 x i32*>* %b
  %vals = call <2 x i32> @llvm.masked.gather.v2i32(<2 x i32*> %ptrs, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i32> undef)
  store <2 x i32> %vals, <2 x i32>* %a
  ret void
}

define void @masked_gather_v4i32(<4 x i32>* %a, <4 x i32*>* %b) #0 {
; CHECK-LABEL: masked_gather_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1]
; CHECK-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    uzp1 z0.s, z0.s, z0.s
; CHECK-NEXT:    str q0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <4 x i32*>, <4 x i32*>* %b
  %vals = call <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*> %ptrs, i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> undef)
  store <4 x i32> %vals, <4 x i32>* %a
  ret void
}

define void @masked_gather_v8i32(<8 x i32>* %a, <8 x i32*>* %b) #0 {
; VBITS_EQ_256-LABEL: masked_gather_v8i32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; VBITS_EQ_256-NEXT:    ld1w { z1.d }, p0/z, [z1.d]
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl4
; VBITS_EQ_256-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_EQ_256-NEXT:    uzp1 z1.s, z1.s, z1.s
; VBITS_EQ_256-NEXT:    splice z1.s, p0, z1.s, z0.s
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_gather_v8i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %ptrs = load <8 x i32*>, <8 x i32*>* %b
  %vals = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %ptrs, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  store <8 x i32> %vals, <8 x i32>* %a
  ret void
}

define void @masked_gather_v16i32(<16 x i32>* %a, <16 x i32*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_gather_v16i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %ptrs = load <16 x i32*>, <16 x i32*>* %b
  %vals = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %ptrs, i32 8, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x i32> undef)
  store <16 x i32> %vals, <16 x i32>* %a
  ret void
}

define void @masked_gather_v32i32(<32 x i32>* %a, <32 x i32*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_gather_v32i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    ld1w { z0.d }, p0/z, [z0.d]
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    uzp1 z0.s, z0.s, z0.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %ptrs = load <32 x i32*>, <32 x i32*>* %b
  %vals = call <32 x i32> @llvm.masked.gather.v32i32(<32 x i32*> %ptrs, i32 8, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i32> undef)
  store <32 x i32> %vals, <32 x i32>* %a
  ret void
}

;
; LD1D
;

define void @masked_gather_v2i64(<2 x i64>* %a, <2 x i64*>* %b) #0 {
; CHECK-LABEL: masked_gather_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x1]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    str q0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <2 x i64*>, <2 x i64*>* %b
  %vals = call <2 x i64> @llvm.masked.gather.v2i64(<2 x i64*> %ptrs, i32 8, <2 x i1> <i1 true, i1 true>, <2 x i64> undef)
  store <2 x i64> %vals, <2 x i64>* %a
  ret void
}

define void @masked_gather_v4i64(<4 x i64>* %a, <4 x i64*>* %b) #0 {
; CHECK-LABEL: masked_gather_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x1]
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %ptrs = load <4 x i64*>, <4 x i64*>* %b
  %vals = call <4 x i64> @llvm.masked.gather.v4i64(<4 x i64*> %ptrs, i32 8, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i64> undef)
  store <4 x i64> %vals, <4 x i64>* %a
  ret void
}

define void @masked_gather_v8i64(<8 x i64>* %a, <8 x i64*>* %b) #0 {
; VBITS_EQ_256-LABEL: masked_gather_v8i64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [z1.d]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p0, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_gather_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %ptrs = load <8 x i64*>, <8 x i64*>* %b
  %vals = call <8 x i64> @llvm.masked.gather.v8i64(<8 x i64*> %ptrs, i32 8, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i64> undef)
  store <8 x i64> %vals, <8 x i64>* %a
  ret void
}

define void @masked_gather_v16i64(<16 x i64>* %a, <16 x i64*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_gather_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %ptrs = load <16 x i64*>, <16 x i64*>* %b
  %vals = call <16 x i64> @llvm.masked.gather.v16i64(<16 x i64*> %ptrs, i32 8, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x i64> undef)
  store <16 x i64> %vals, <16 x i64>* %a
  ret void
}

define void @masked_gather_v32i64(<32 x i64>* %a, <32 x i64*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_gather_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [z0.d]
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %ptrs = load <32 x i64*>, <32 x i64*>* %b
  %vals = call <32 x i64> @llvm.masked.gather.v32i64(<32 x i64*> %ptrs, i32 8, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true,
                                                                                          i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i64> undef)
  store <32 x i64> %vals, <32 x i64>* %a
  ret void
}

declare <2 x i8> @llvm.masked.gather.v2i8(<2 x i8*>, i32, <2 x i1>, <2 x i8>)
declare <4 x i8> @llvm.masked.gather.v4i8(<4 x i8*>, i32, <4 x i1>, <4 x i8>)
declare <8 x i8> @llvm.masked.gather.v8i8(<8 x i8*>, i32, <8 x i1>, <8 x i8>)
declare <16 x i8> @llvm.masked.gather.v16i8(<16 x i8*>, i32, <16 x i1>, <16 x i8>)
declare <32 x i8> @llvm.masked.gather.v32i8(<32 x i8*>, i32, <32 x i1>, <32 x i8>)

declare <2 x i16> @llvm.masked.gather.v2i16(<2 x i16*>, i32, <2 x i1>, <2 x i16>)
declare <4 x i16> @llvm.masked.gather.v4i16(<4 x i16*>, i32, <4 x i1>, <4 x i16>)
declare <8 x i16> @llvm.masked.gather.v8i16(<8 x i16*>, i32, <8 x i1>, <8 x i16>)
declare <16 x i16> @llvm.masked.gather.v16i16(<16 x i16*>, i32, <16 x i1>, <16 x i16>)
declare <32 x i16> @llvm.masked.gather.v32i16(<32 x i16*>, i32, <32 x i1>, <32 x i16>)

declare <2 x i32> @llvm.masked.gather.v2i32(<2 x i32*>, i32, <2 x i1>, <2 x i32>)
declare <4 x i32> @llvm.masked.gather.v4i32(<4 x i32*>, i32, <4 x i1>, <4 x i32>)
declare <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*>, i32, <8 x i1>, <8 x i32>)
declare <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*>, i32, <16 x i1>, <16 x i32>)
declare <32 x i32> @llvm.masked.gather.v32i32(<32 x i32*>, i32, <32 x i1>, <32 x i32>)

declare <2 x i64> @llvm.masked.gather.v2i64(<2 x i64*>, i32, <2 x i1>, <2 x i64>)
declare <4 x i64> @llvm.masked.gather.v4i64(<4 x i64*>, i32, <4 x i1>, <4 x i64>)
declare <8 x i64> @llvm.masked.gather.v8i64(<8 x i64*>, i32, <8 x i1>, <8 x i64>)
declare <16 x i64> @llvm.masked.gather.v16i64(<16 x i64*>, i32, <16 x i1>, <16 x i64>)
declare <32 x i64> @llvm.masked.gather.v32i64(<32 x i64*>, i32, <32 x i1>, <32 x i64>)

attributes #0 = { "target-features"="+sve" }
