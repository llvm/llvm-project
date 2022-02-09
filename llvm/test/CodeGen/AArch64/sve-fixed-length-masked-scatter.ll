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
; ST1B
;

define void @masked_scatter_v2i8(<2 x i8>* %a, <2 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldrb w8, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    ldrb w8, [x0, #1]
; CHECK-NEXT:    mov v0.s[1], w8
; CHECK-NEXT:    cmeq v1.2s, v0.2s, #0
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    sshll v1.2d, v1.2s, #0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1b { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x i8>, <2 x i8>* %a
  %ptrs = load <2 x i8*>, <2 x i8*>* %b
  %mask = icmp eq <2 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i8(<2 x i8> %vals, <2 x i8*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i8(<4 x i8>* %a, <4 x i8*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    ushll v0.8h, v0.8b, #0
; CHECK-NEXT:    cmeq v2.4h, v0.4h, #0
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    sunpklo z2.s, z2.h
; CHECK-NEXT:    sunpklo z2.d, z2.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; CHECK-NEXT:    st1b { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x i8>, <4 x i8>* %a
  %ptrs = load <4 x i8*>, <4 x i8*>* %b
  %mask = icmp eq <4 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i8(<4 x i8> %vals, <4 x i8*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i8(<8 x i8>* %a, <8 x i8*>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: masked_scatter_v8i8:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ldr d0, [x0]
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    cmeq v1.8b, v0.8b, #0
; VBITS_EQ_256-NEXT:    zip1 v5.8b, v0.8b, v0.8b
; VBITS_EQ_256-NEXT:    ld1d { z3.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z4.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    zip1 v2.8b, v1.8b, v0.8b
; VBITS_EQ_256-NEXT:    zip2 v1.8b, v1.8b, v0.8b
; VBITS_EQ_256-NEXT:    zip2 v0.8b, v0.8b, v0.8b
; VBITS_EQ_256-NEXT:    shl v2.4h, v2.4h, #8
; VBITS_EQ_256-NEXT:    shl v1.4h, v1.4h, #8
; VBITS_EQ_256-NEXT:    uunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    uunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    sshr v2.4h, v2.4h, #8
; VBITS_EQ_256-NEXT:    sshr v1.4h, v1.4h, #8
; VBITS_EQ_256-NEXT:    sunpklo z2.s, z2.h
; VBITS_EQ_256-NEXT:    sunpklo z1.s, z1.h
; VBITS_EQ_256-NEXT:    sunpklo z2.d, z2.s
; VBITS_EQ_256-NEXT:    sunpklo z1.d, z1.s
; VBITS_EQ_256-NEXT:    cmpne p1.d, p0/z, z2.d, #0
; VBITS_EQ_256-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; VBITS_EQ_256-NEXT:    uunpklo z1.s, z5.h
; VBITS_EQ_256-NEXT:    uunpklo z1.d, z1.s
; VBITS_EQ_256-NEXT:    st1b { z1.d }, p1, [z4.d]
; VBITS_EQ_256-NEXT:    st1b { z0.d }, p0, [z3.d]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_scatter_v8i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr d0, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmeq v2.8b, v0.8b, #0
; VBITS_GE_512-NEXT:    uunpklo z0.h, z0.b
; VBITS_GE_512-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    sunpklo z2.h, z2.b
; VBITS_GE_512-NEXT:    sunpklo z2.s, z2.h
; VBITS_GE_512-NEXT:    sunpklo z2.d, z2.s
; VBITS_GE_512-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; VBITS_GE_512-NEXT:    st1b { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x i8>, <8 x i8>* %a
  %ptrs = load <8 x i8*>, <8 x i8*>* %b
  %mask = icmp eq <8 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i8(<8 x i8> %vals, <8 x i8*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i8(<16 x i8>* %a, <16 x i8*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ldr q0, [x0]
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    cmeq v2.16b, v0.16b, #0
; VBITS_GE_1024-NEXT:    uunpklo z0.h, z0.b
; VBITS_GE_1024-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    sunpklo z2.h, z2.b
; VBITS_GE_1024-NEXT:    sunpklo z2.s, z2.h
; VBITS_GE_1024-NEXT:    sunpklo z2.d, z2.s
; VBITS_GE_1024-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; VBITS_GE_1024-NEXT:    st1b { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x i8>, <16 x i8>* %a
  %ptrs = load <16 x i8*>, <16 x i8*>* %b
  %mask = icmp eq <16 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i8(<16 x i8> %vals, <16 x i8*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i8(<32 x i8>* %a, <32 x i8*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.b, p0/z, z0.b, #0
; VBITS_GE_2048-NEXT:    uunpklo z0.h, z0.b
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1b { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x i8>, <32 x i8>* %a
  %ptrs = load <32 x i8*>, <32 x i8*>* %b
  %mask = icmp eq <32 x i8> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i8(<32 x i8> %vals, <32 x i8*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1H
;

define void @masked_scatter_v2i16(<2 x i16>* %a, <2 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldrh w8, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    fmov s0, w8
; CHECK-NEXT:    ldrh w8, [x0, #2]
; CHECK-NEXT:    mov v0.s[1], w8
; CHECK-NEXT:    cmeq v1.2s, v0.2s, #0
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    sshll v1.2d, v1.2s, #0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1h { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x i16>, <2 x i16>* %a
  %ptrs = load <2 x i16*>, <2 x i16*>* %b
  %mask = icmp eq <2 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i16(<2 x i16> %vals, <2 x i16*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i16(<4 x i16>* %a, <4 x i16*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    cmeq v2.4h, v0.4h, #0
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    sunpklo z2.s, z2.h
; CHECK-NEXT:    sunpklo z2.d, z2.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; CHECK-NEXT:    st1h { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x i16>, <4 x i16>* %a
  %ptrs = load <4 x i16*>, <4 x i16*>* %b
  %mask = icmp eq <4 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i16(<4 x i16> %vals, <4 x i16*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i16(<8 x i16>* %a, <8 x i16*>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: masked_scatter_v8i16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ldr q0, [x0]
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    cmeq v1.8h, v0.8h, #0
; VBITS_EQ_256-NEXT:    ld1d { z4.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ext v3.16b, v0.16b, v0.16b, #8
; VBITS_EQ_256-NEXT:    uunpklo z0.s, z0.h
; VBITS_EQ_256-NEXT:    uunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    sunpklo z2.s, z1.h
; VBITS_EQ_256-NEXT:    ext v1.16b, v1.16b, v1.16b, #8
; VBITS_EQ_256-NEXT:    sunpklo z2.d, z2.s
; VBITS_EQ_256-NEXT:    cmpne p1.d, p0/z, z2.d, #0
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    uunpklo z3.s, z3.h
; VBITS_EQ_256-NEXT:    sunpklo z1.s, z1.h
; VBITS_EQ_256-NEXT:    sunpklo z1.d, z1.s
; VBITS_EQ_256-NEXT:    st1h { z0.d }, p1, [z2.d]
; VBITS_EQ_256-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; VBITS_EQ_256-NEXT:    uunpklo z1.d, z3.s
; VBITS_EQ_256-NEXT:    st1h { z1.d }, p0, [z4.d]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_scatter_v8i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmeq v2.8h, v0.8h, #0
; VBITS_GE_512-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    sunpklo z2.s, z2.h
; VBITS_GE_512-NEXT:    sunpklo z2.d, z2.s
; VBITS_GE_512-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; VBITS_GE_512-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x i16>, <8 x i16>* %a
  %ptrs = load <8 x i16*>, <8 x i16*>* %b
  %mask = icmp eq <8 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i16(<8 x i16> %vals, <8 x i16*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i16(<16 x i16>* %a, <16 x i16*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    ptrue p1.d, vl16
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_1024-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_1024-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x i16>, <16 x i16>* %a
  %ptrs = load <16 x i16*>, <16 x i16*>* %b
  %mask = icmp eq <16 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i16(<16 x i16> %vals, <16 x i16*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i16(<32 x i16>* %a, <32 x i16*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.h, p0/z, z0.h, #0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x i16>, <32 x i16>* %a
  %ptrs = load <32 x i16*>, <32 x i16*>* %b
  %mask = icmp eq <32 x i16> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i16(<32 x i16> %vals, <32 x i16*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1W
;

define void @masked_scatter_v2i32(<2 x i32>* %a, <2 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    cmeq v1.2s, v0.2s, #0
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    sshll v1.2d, v1.2s, #0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1w { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x i32>, <2 x i32>* %a
  %ptrs = load <2 x i32*>, <2 x i32*>* %b
  %mask = icmp eq <2 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i32(<2 x i32> %vals, <2 x i32*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i32(<4 x i32>* %a, <4 x i32*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    cmeq v2.4s, v0.4s, #0
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    sunpklo z2.d, z2.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; CHECK-NEXT:    st1w { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x i32>, <4 x i32>* %a
  %ptrs = load <4 x i32*>, <4 x i32*>* %b
  %mask = icmp eq <4 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i32(<4 x i32> %vals, <4 x i32*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i32(<8 x i32>* %a, <8 x i32*>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: masked_scatter_v8i32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_EQ_256-NEXT:    punpklo p1.h, p0.b
; VBITS_EQ_256-NEXT:    mov z4.s, p0/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    mov z1.d, p1/z, #-1 // =0xffffffffffffffff
; VBITS_EQ_256-NEXT:    ptrue p1.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p1/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z3.d }, p1/z, [x1]
; VBITS_EQ_256-NEXT:    ext z4.b, z4.b, z4.b, #16
; VBITS_EQ_256-NEXT:    cmpne p0.d, p1/z, z1.d, #0
; VBITS_EQ_256-NEXT:    uunpklo z1.d, z0.s
; VBITS_EQ_256-NEXT:    sunpklo z4.d, z4.s
; VBITS_EQ_256-NEXT:    ext z0.b, z0.b, z0.b, #16
; VBITS_EQ_256-NEXT:    cmpne p1.d, p1/z, z4.d, #0
; VBITS_EQ_256-NEXT:    uunpklo z0.d, z0.s
; VBITS_EQ_256-NEXT:    st1w { z1.d }, p0, [z3.d]
; VBITS_EQ_256-NEXT:    st1w { z0.d }, p1, [z2.d]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_scatter_v8i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ptrue p1.d, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x i32>, <8 x i32>* %a
  %ptrs = load <8 x i32*>, <8 x i32*>* %b
  %mask = icmp eq <8 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i32(<8 x i32> %vals, <8 x i32*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i32(<16 x i32>* %a, <16 x i32*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ptrue p1.d, vl16
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_1024-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x i32>, <16 x i32>* %a
  %ptrs = load <16 x i32*>, <16 x i32*>* %b
  %mask = icmp eq <16 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i32(<16 x i32> %vals, <16 x i32*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i32(<32 x i32>* %a, <32 x i32*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.s, p0/z, z0.s, #0
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x i32>, <32 x i32>* %a
  %ptrs = load <32 x i32*>, <32 x i32*>* %b
  %mask = icmp eq <32 x i32> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i32(<32 x i32> %vals, <32 x i32*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1D
;

; Scalarize 1 x i64 scatters
define void @masked_scatter_v1i64(<1 x i64>* %a, <1 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v1i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    cbnz x8, .LBB15_2
; CHECK-NEXT:  // %bb.1: // %cond.store
; CHECK-NEXT:    ldr d1, [x1]
; CHECK-NEXT:    fmov x8, d1
; CHECK-NEXT:    str d0, [x8]
; CHECK-NEXT:  .LBB15_2: // %else
; CHECK-NEXT:    ret
  %vals = load <1 x i64>, <1 x i64>* %a
  %ptrs = load <1 x i64*>, <1 x i64*>* %b
  %mask = icmp eq <1 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v1i64(<1 x i64> %vals, <1 x i64*> %ptrs, i32 8, <1 x i1> %mask)
  ret void
}

define void @masked_scatter_v2i64(<2 x i64>* %a, <2 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    cmeq v1.2d, v0.2d, #0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1d { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x i64>, <2 x i64>* %a
  %ptrs = load <2 x i64*>, <2 x i64*>* %b
  %mask = icmp eq <2 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2i64(<2 x i64> %vals, <2 x i64*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4i64(<4 x i64>* %a, <4 x i64*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    cmpeq p0.d, p0/z, z0.d, #0
; CHECK-NEXT:    st1d { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x i64>, <4 x i64>* %a
  %ptrs = load <4 x i64*>, <4 x i64*>* %b
  %mask = icmp eq <4 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4i64(<4 x i64> %vals, <4 x i64*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8i64(<8 x i64>* %a, <8 x i64*>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: masked_scatter_v8i64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x0]
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z3.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    cmpeq p1.d, p0/z, z0.d, #0
; VBITS_EQ_256-NEXT:    cmpeq p0.d, p0/z, z1.d, #0
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [z3.d]
; VBITS_EQ_256-NEXT:    st1d { z0.d }, p1, [z2.d]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: masked_scatter_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    cmpeq p0.d, p0/z, z0.d, #0
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x i64>, <8 x i64>* %a
  %ptrs = load <8 x i64*>, <8 x i64*>* %b
  %mask = icmp eq <8 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8i64(<8 x i64> %vals, <8 x i64*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16i64(<16 x i64>* %a, <16 x i64*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    cmpeq p0.d, p0/z, z0.d, #0
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x i64>, <16 x i64>* %a
  %ptrs = load <16 x i64*>, <16 x i64*>* %b
  %mask = icmp eq <16 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16i64(<16 x i64> %vals, <16 x i64*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32i64(<32 x i64>* %a, <32 x i64*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    cmpeq p0.d, p0/z, z0.d, #0
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x i64>, <32 x i64>* %a
  %ptrs = load <32 x i64*>, <32 x i64*>* %b
  %mask = icmp eq <32 x i64> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32i64(<32 x i64> %vals, <32 x i64*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1H (float)
;

define void @masked_scatter_v2f16(<2 x half>* %a, <2 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s1, [x0]
; CHECK-NEXT:    movi d0, #0000000000000000
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    fcmeq v2.4h, v1.4h, #0.0
; CHECK-NEXT:    uunpklo z1.s, z1.h
; CHECK-NEXT:    umov w8, v2.h[0]
; CHECK-NEXT:    umov w9, v2.h[1]
; CHECK-NEXT:    fmov s2, w8
; CHECK-NEXT:    mov v2.s[1], w9
; CHECK-NEXT:    shl v2.2s, v2.2s, #16
; CHECK-NEXT:    sshr v2.2s, v2.2s, #16
; CHECK-NEXT:    fmov w8, s2
; CHECK-NEXT:    mov w9, v2.s[1]
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    mov v0.h[0], w8
; CHECK-NEXT:    mov v0.h[1], w9
; CHECK-NEXT:    shl v0.4h, v0.4h, #15
; CHECK-NEXT:    cmlt v0.4h, v0.4h, #0
; CHECK-NEXT:    sunpklo z0.s, z0.h
; CHECK-NEXT:    sunpklo z0.d, z0.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z0.d, #0
; CHECK-NEXT:    uunpklo z0.d, z1.s
; CHECK-NEXT:    st1h { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x half>, <2 x half>* %a
  %ptrs = load <2 x half*>, <2 x half*>* %b
  %mask = fcmp oeq <2 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f16(<2 x half> %vals, <2 x half*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f16(<4 x half>* %a, <4 x half*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    fcmeq v2.4h, v0.4h, #0.0
; CHECK-NEXT:    uunpklo z0.s, z0.h
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    sunpklo z2.s, z2.h
; CHECK-NEXT:    sunpklo z2.d, z2.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; CHECK-NEXT:    st1h { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x half>, <4 x half>* %a
  %ptrs = load <4 x half*>, <4 x half*>* %b
  %mask = fcmp oeq <4 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f16(<4 x half> %vals, <4 x half*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f16(<8 x half>* %a, <8 x half*>* %b) #0 {
; VBITS_GE_512-LABEL: masked_scatter_v8f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ldr q0, [x0]
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq v2.8h, v0.8h, #0.0
; VBITS_GE_512-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    sunpklo z2.s, z2.h
; VBITS_GE_512-NEXT:    sunpklo z2.d, z2.s
; VBITS_GE_512-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; VBITS_GE_512-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x half>, <8 x half>* %a
  %ptrs = load <8 x half*>, <8 x half*>* %b
  %mask = fcmp oeq <8 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f16(<8 x half> %vals, <8 x half*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f16(<16 x half>* %a, <16 x half*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl16
; VBITS_GE_1024-NEXT:    ptrue p1.d, vl16
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_1024-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x half>, <16 x half>* %a
  %ptrs = load <16 x half*>, <16 x half*>* %b
  %mask = fcmp oeq <16 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f16(<16 x half> %vals, <16 x half*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f16(<32 x half>* %a, <32 x half*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x half>, <32 x half>* %a
  %ptrs = load <32 x half*>, <32 x half*>* %b
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1W (float)
;

define void @masked_scatter_v2f32(<2 x float>* %a, <2 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    fcmeq v1.2s, v0.2s, #0.0
; CHECK-NEXT:    ushll v0.2d, v0.2s, #0
; CHECK-NEXT:    sshll v1.2d, v1.2s, #0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1w { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x float>, <2 x float>* %a
  %ptrs = load <2 x float*>, <2 x float*>* %b
  %mask = fcmp oeq <2 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f32(<2 x float> %vals, <2 x float*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f32(<4 x float>* %a, <4 x float*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    fcmeq v2.4s, v0.4s, #0.0
; CHECK-NEXT:    uunpklo z0.d, z0.s
; CHECK-NEXT:    sunpklo z2.d, z2.s
; CHECK-NEXT:    cmpne p0.d, p0/z, z2.d, #0
; CHECK-NEXT:    st1w { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x float>, <4 x float>* %a
  %ptrs = load <4 x float*>, <4 x float*>* %b
  %mask = fcmp oeq <4 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f32(<4 x float> %vals, <4 x float*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f32(<8 x float>* %a, <8 x float*>* %b) #0 {
; VBITS_GE_512-LABEL: masked_scatter_v8f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl8
; VBITS_GE_512-NEXT:    ptrue p1.d, vl8
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_512-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_512-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_512-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x float>, <8 x float>* %a
  %ptrs = load <8 x float*>, <8 x float*>* %b
  %mask = fcmp oeq <8 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f32(<8 x float> %vals, <8 x float*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f32(<16 x float>* %a, <16 x float*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl16
; VBITS_GE_1024-NEXT:    ptrue p1.d, vl16
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_1024-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_1024-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_1024-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x float>, <16 x float>* %a
  %ptrs = load <16 x float*>, <16 x float*>* %b
  %mask = fcmp oeq <16 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f32(<16 x float> %vals, <16 x float*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f32(<32 x float>* %a, <32 x float*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %ptrs = load <32 x float*>, <32 x float*>* %b
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

;
; ST1D (float)
;

; Scalarize 1 x double scatters
define void @masked_scatter_v1f64(<1 x double>* %a, <1 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v1f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    fcmp d0, #0.0
; CHECK-NEXT:    b.ne .LBB31_2
; CHECK-NEXT:  // %bb.1: // %cond.store
; CHECK-NEXT:    ldr d1, [x1]
; CHECK-NEXT:    fmov x8, d1
; CHECK-NEXT:    str d0, [x8]
; CHECK-NEXT:  .LBB31_2: // %else
; CHECK-NEXT:    ret
  %vals = load <1 x double>, <1 x double>* %a
  %ptrs = load <1 x double*>, <1 x double*>* %b
  %mask = fcmp oeq <1 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v1f64(<1 x double> %vals, <1 x double*> %ptrs, i32 8, <1 x i1> %mask)
  ret void
}

define void @masked_scatter_v2f64(<2 x double>* %a, <2 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.d, vl2
; CHECK-NEXT:    ldr q2, [x1]
; CHECK-NEXT:    fcmeq v1.2d, v0.2d, #0.0
; CHECK-NEXT:    cmpne p0.d, p0/z, z1.d, #0
; CHECK-NEXT:    st1d { z0.d }, p0, [z2.d]
; CHECK-NEXT:    ret
  %vals = load <2 x double>, <2 x double>* %a
  %ptrs = load <2 x double*>, <2 x double*>* %b
  %mask = fcmp oeq <2 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v2f64(<2 x double> %vals, <2 x double*> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_scatter_v4f64(<4 x double>* %a, <4 x double*>* %b) #0 {
; CHECK-LABEL: masked_scatter_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; CHECK-NEXT:    st1d { z0.d }, p0, [z1.d]
; CHECK-NEXT:    ret
  %vals = load <4 x double>, <4 x double>* %a
  %ptrs = load <4 x double*>, <4 x double*>* %b
  %mask = fcmp oeq <4 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v4f64(<4 x double> %vals, <4 x double*> %ptrs, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_scatter_v8f64(<8 x double>* %a, <8 x double*>* %b) #0 {
; VBITS_GE_512-LABEL: masked_scatter_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:    ret
  %vals = load <8 x double>, <8 x double>* %a
  %ptrs = load <8 x double*>, <8 x double*>* %b
  %mask = fcmp oeq <8 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f64(<8 x double> %vals, <8 x double*> %ptrs, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_scatter_v16f64(<16 x double>* %a, <16 x double*>* %b) #0 {
; VBITS_GE_1024-LABEL: masked_scatter_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; VBITS_GE_1024-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_1024-NEXT:    ret
  %vals = load <16 x double>, <16 x double>* %a
  %ptrs = load <16 x double*>, <16 x double*>* %b
  %mask = fcmp oeq <16 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v16f64(<16 x double> %vals, <16 x double*> %ptrs, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_scatter_v32f64(<32 x double>* %a, <32 x double*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x double>, <32 x double>* %a
  %ptrs = load <32 x double*>, <32 x double*>* %b
  %mask = fcmp oeq <32 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f64(<32 x double> %vals, <32 x double*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; The above tests test the types, the below tests check that the addressing
; modes still function

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_scaled_sext_f16(<32 x half>* %a, <32 x i32>* %b, half* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_scaled_sext_f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1sw { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [x2, z1.d, lsl #1]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr half, half* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_scaled_sext_f32(<32 x float>* %a, <32 x i32>* %b, float* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_scaled_sext_f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1sw { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [x2, z1.d, lsl #2]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr float, float* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_scaled_sext_f64(<32 x double>* %a, <32 x i32>* %b, double* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_scaled_sext_f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1sw { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; VBITS_GE_2048-NEXT:    st1d { z0.d }, p0, [x2, z1.d, lsl #3]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x double>, <32 x double>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr double, double* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f64(<32 x double> %vals, <32 x double*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_scaled_zext(<32 x half>* %a, <32 x i32>* %b, half* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_scaled_zext:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [x2, z1.d, lsl #1]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = zext <32 x i32> %idxs to <32 x i64>
  %ptrs = getelementptr half, half* %base, <32 x i64> %ext
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_unscaled_sext(<32 x half>* %a, <32 x i32>* %b, i8* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_unscaled_sext:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1sw { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [x2, z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = sext <32 x i32> %idxs to <32 x i64>
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %ext
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x half*>
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; NOTE: This produces an non-optimal addressing mode due to a temporary workaround
define void @masked_scatter_32b_unscaled_zext(<32 x half>* %a, <32 x i32>* %b, i8* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_32b_unscaled_zext:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.h, p0/z, z0.h, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.s, z0.h
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1h { z0.d }, p0, [x2, z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x half>, <32 x half>* %a
  %idxs = load <32 x i32>, <32 x i32>* %b
  %ext = zext <32 x i32> %idxs to <32 x i64>
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %ext
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x half*>
  %mask = fcmp oeq <32 x half> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f16(<32 x half> %vals, <32 x half*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_64b_scaled(<32 x float>* %a, <32 x i64>* %b, float* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_64b_scaled:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [x2, z1.d, lsl #2]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i64>, <32 x i64>* %b
  %ptrs = getelementptr float, float* %base, <32 x i64> %idxs
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_scatter_64b_unscaled(<32 x float>* %a, <32 x i64>* %b, i8* %base) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_64b_unscaled:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [x2, z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %idxs = load <32 x i64>, <32 x i64>* %b
  %byte_ptrs = getelementptr i8, i8* %base, <32 x i64> %idxs
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; FIXME: This case does not yet codegen well due to deficiencies in opcode selection
define void @masked_scatter_vec_plus_reg(<32 x float>* %a, <32 x i8*>* %b, i64 %off) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_vec_plus_reg:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    mov z2.d, x2
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    add z1.d, z1.d, z2.d
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %bases = load <32 x i8*>, <32 x i8*>* %b
  %byte_ptrs = getelementptr i8, <32 x i8*> %bases, i64 %off
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; FIXME: This case does not yet codegen well due to deficiencies in opcode selection
define void @masked_scatter_vec_plus_imm(<32 x float>* %a, <32 x i8*>* %b) #0 {
; VBITS_GE_2048-LABEL: masked_scatter_vec_plus_imm:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl32
; VBITS_GE_2048-NEXT:    ptrue p1.d, vl32
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p1/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, #0.0
; VBITS_GE_2048-NEXT:    add z1.d, z1.d, #4
; VBITS_GE_2048-NEXT:    uunpklo z0.d, z0.s
; VBITS_GE_2048-NEXT:    punpklo p0.h, p0.b
; VBITS_GE_2048-NEXT:    st1w { z0.d }, p0, [z1.d]
; VBITS_GE_2048-NEXT:    ret
  %vals = load <32 x float>, <32 x float>* %a
  %bases = load <32 x i8*>, <32 x i8*>* %b
  %byte_ptrs = getelementptr i8, <32 x i8*> %bases, i64 4
  %ptrs = bitcast <32 x i8*> %byte_ptrs to <32 x float*>
  %mask = fcmp oeq <32 x float> %vals, zeroinitializer
  call void @llvm.masked.scatter.v32f32(<32 x float> %vals, <32 x float*> %ptrs, i32 8, <32 x i1> %mask)
  ret void
}

; extract_subvec(...(insert_subvec(a,b,c))) -> extract_subvec(bitcast(b),d) like
; combines can effectively unlegalise bitcast operations. This test ensures such
; combines do not happen after operation legalisation. When not prevented the
; test triggers infinite combine->legalise->combine->...
;
; NOTE: For this test to function correctly it's critical for %vals to be in a
; different block to the scatter store.  If not, the problematic bitcast will be
; removed before operation legalisation and thus not exercise the combine.
define void @masked_scatter_bitcast_infinite_loop(<8 x double>* %a, <8 x double*>* %b, i1 %cond) #0 {
; VBITS_GE_512-LABEL: masked_scatter_bitcast_infinite_loop:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    tbz w2, #0, .LBB47_2
; VBITS_GE_512-NEXT:  // %bb.1: // %bb.1
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p0.d, p0/z, z0.d, #0.0
; VBITS_GE_512-NEXT:    st1d { z0.d }, p0, [z1.d]
; VBITS_GE_512-NEXT:  .LBB47_2: // %bb.2
; VBITS_GE_512-NEXT:    ret
  %vals = load volatile <8 x double>, <8 x double>* %a
  br i1 %cond, label %bb.1, label %bb.2

bb.1:
  %ptrs = load <8 x double*>, <8 x double*>* %b
  %mask = fcmp oeq <8 x double> %vals, zeroinitializer
  call void @llvm.masked.scatter.v8f64(<8 x double> %vals, <8 x double*> %ptrs, i32 8, <8 x i1> %mask)
  br label %bb.2

bb.2:
  ret void
}

declare void @llvm.masked.scatter.v2i8(<2 x i8>, <2 x i8*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i8(<4 x i8>, <4 x i8*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i8(<8 x i8>, <8 x i8*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i8(<16 x i8>, <16 x i8*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i8(<32 x i8>, <32 x i8*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2i16(<2 x i16>, <2 x i16*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i16(<4 x i16>, <4 x i16*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i16(<8 x i16>, <8 x i16*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i16(<16 x i16>, <16 x i16*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i16(<32 x i16>, <32 x i16*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2i32(<2 x i32>, <2 x i32*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i32(<4 x i32>, <4 x i32*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i32(<8 x i32>, <8 x i32*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i32(<16 x i32>, <16 x i32*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i32(<32 x i32>, <32 x i32*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v1i64(<1 x i64>, <1 x i64*>, i32, <1 x i1>)
declare void @llvm.masked.scatter.v2i64(<2 x i64>, <2 x i64*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4i64(<4 x i64>, <4 x i64*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8i64(<8 x i64>, <8 x i64*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16i64(<16 x i64>, <16 x i64*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32i64(<32 x i64>, <32 x i64*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2f16(<2 x half>, <2 x half*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f16(<4 x half>, <4 x half*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f16(<8 x half>, <8 x half*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f16(<16 x half>, <16 x half*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f16(<32 x half>, <32 x half*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v2f32(<2 x float>, <2 x float*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f32(<4 x float>, <4 x float*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f32(<8 x float>, <8 x float*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f32(<16 x float>, <16 x float*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f32(<32 x float>, <32 x float*>, i32, <32 x i1>)

declare void @llvm.masked.scatter.v1f64(<1 x double>, <1 x double*>, i32, <1 x i1>)
declare void @llvm.masked.scatter.v2f64(<2 x double>, <2 x double*>, i32, <2 x i1>)
declare void @llvm.masked.scatter.v4f64(<4 x double>, <4 x double*>, i32, <4 x i1>)
declare void @llvm.masked.scatter.v8f64(<8 x double>, <8 x double*>, i32, <8 x i1>)
declare void @llvm.masked.scatter.v16f64(<16 x double>, <16 x double*>, i32, <16 x i1>)
declare void @llvm.masked.scatter.v32f64(<32 x double>, <32 x double*>, i32, <32 x i1>)

attributes #0 = { "target-features"="+sve" }
