; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -D#VBYTES=16  -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -D#VBYTES=32  -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -D#VBYTES=64  -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -D#VBYTES=128 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -D#VBYTES=256 -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;;
;; Masked Stores
;;
define void @masked_store_v2f16(<2 x half>* %ap, <2 x half>* %bp) #0 {
; CHECK-LABEL: masked_store_v2f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr s1, [x0]
; CHECK-NEXT:    movi d0, #0000000000000000
; CHECK-NEXT:    ldr s2, [x1]
; CHECK-NEXT:    ptrue p0.h, vl4
; CHECK-NEXT:    fcmeq v2.4h, v1.4h, v2.4h
; CHECK-NEXT:    umov w8, v2.h[0]
; CHECK-NEXT:    umov w9, v2.h[1]
; CHECK-NEXT:    fmov s2, w8
; CHECK-NEXT:    mov v2.s[1], w9
; CHECK-NEXT:    shl v2.2s, v2.2s, #16
; CHECK-NEXT:    sshr v2.2s, v2.2s, #16
; CHECK-NEXT:    fmov w8, s2
; CHECK-NEXT:    mov w9, v2.s[1]
; CHECK-NEXT:    mov v0.h[0], w8
; CHECK-NEXT:    mov v0.h[1], w9
; CHECK-NEXT:    shl v0.4h, v0.4h, #15
; CHECK-NEXT:    cmlt v0.4h, v0.4h, #0
; CHECK-NEXT:    cmpne p0.h, p0/z, z0.h, #0
; CHECK-NEXT:    st1h { z1.h }, p0, [x1]
; CHECK-NEXT:    ret
  %a = load <2 x half>, <2 x half>* %ap
  %b = load <2 x half>, <2 x half>* %bp
  %mask = fcmp oeq <2 x half> %a, %b
  call void @llvm.masked.store.v2f16(<2 x half> %a, <2 x half>* %bp, i32 8, <2 x i1> %mask)
  ret void
}


define void @masked_store_v2f32(<2 x float>* %ap, <2 x float>* %bp) #0 {
; CHECK-LABEL: masked_store_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl2
; CHECK-NEXT:    ldr d1, [x1]
; CHECK-NEXT:    fcmeq v1.2s, v0.2s, v1.2s
; CHECK-NEXT:    cmpne p0.s, p0/z, z1.s, #0
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %a = load <2 x float>, <2 x float>* %ap
  %b = load <2 x float>, <2 x float>* %bp
  %mask = fcmp oeq <2 x float> %a, %b
  call void @llvm.masked.store.v2f32(<2 x float> %a, <2 x float>* %bp, i32 8, <2 x i1> %mask)
  ret void
}

define void @masked_store_v4f32(<4 x float>* %ap, <4 x float>* %bp) #0 {
; CHECK-LABEL: masked_store_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ptrue p0.s, vl4
; CHECK-NEXT:    ldr q1, [x1]
; CHECK-NEXT:    fcmeq v1.4s, v0.4s, v1.4s
; CHECK-NEXT:    cmpne p0.s, p0/z, z1.s, #0
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %a = load <4 x float>, <4 x float>* %ap
  %b = load <4 x float>, <4 x float>* %bp
  %mask = fcmp oeq <4 x float> %a, %b
  call void @llvm.masked.store.v4f32(<4 x float> %a, <4 x float>* %bp, i32 8, <4 x i1> %mask)
  ret void
}

define void @masked_store_v8f32(<8 x float>* %ap, <8 x float>* %bp) #0 {
; CHECK-LABEL: masked_store_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w { z1.s }, p0/z, [x1]
; CHECK-NEXT:    fcmeq p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x1]
; CHECK-NEXT:    ret
  %a = load <8 x float>, <8 x float>* %ap
  %b = load <8 x float>, <8 x float>* %bp
  %mask = fcmp oeq <8 x float> %a, %b
  call void @llvm.masked.store.v8f32(<8 x float> %a, <8 x float>* %bp, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_store_v16f32(<16 x float>* %ap, <16 x float>* %bp) #0 {
; VBITS_GE_512-LABEL: masked_store_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    fcmeq p0.s, p0/z, z0.s, z1.s
; VBITS_GE_512-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %a = load <16 x float>, <16 x float>* %ap
  %b = load <16 x float>, <16 x float>* %bp
  %mask = fcmp oeq <16 x float> %a, %b
  call void @llvm.masked.store.v16f32(<16 x float> %a, <16 x float>* %ap, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_store_v32f32(<32 x float>* %ap, <32 x float>* %bp) #0 {
; VBITS_GE_1024-LABEL: masked_store_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    fcmeq p0.s, p0/z, z0.s, z1.s
; VBITS_GE_1024-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %a = load <32 x float>, <32 x float>* %ap
  %b = load <32 x float>, <32 x float>* %bp
  %mask = fcmp oeq <32 x float> %a, %b
  call void @llvm.masked.store.v32f32(<32 x float> %a, <32 x float>* %ap, i32 8, <32 x i1> %mask)
  ret void
}

define void @masked_store_v64f32(<64 x float>* %ap, <64 x float>* %bp) #0 {
; VBITS_GE_2048-LABEL: masked_store_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    fcmeq p0.s, p0/z, z0.s, z1.s
; VBITS_GE_2048-NEXT:    st1w { z0.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %a = load <64 x float>, <64 x float>* %ap
  %b = load <64 x float>, <64 x float>* %bp
  %mask = fcmp oeq <64 x float> %a, %b
  call void @llvm.masked.store.v64f32(<64 x float> %a, <64 x float>* %ap, i32 8, <64 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v8i64i8(<8 x i64>* %ap, <8 x i64>* %bp, <8 x i8>* %dest) #0 {
; VBITS_GE_512-LABEL: masked_store_trunc_v8i64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].d, p[[P0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: st1b { [[Z0]].d }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret

  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %val = trunc <8 x i64> %a to <8 x i8>
  call void @llvm.masked.store.v8i8(<8 x i8> %val, <8 x i8>* %dest, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v8i64i16(<8 x i64>* %ap, <8 x i64>* %bp, <8 x i16>* %dest) #0 {
; CHECK-LABEL: masked_store_trunc_v8i64i16:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].d, p[[P0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: st1h { [[Z0]].d }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %val = trunc <8 x i64> %a to <8 x i16>
  call void @llvm.masked.store.v8i16(<8 x i16> %val, <8 x i16>* %dest, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v8i64i32(<8 x i64>* %ap, <8 x i64>* %bp, <8 x i32>* %dest) #0 {
; CHECK-LABEL: masked_store_trunc_v8i64i32:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[Z0:z[0-9]+]].d }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[Z1:z[0-9]+]].d }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].d, p[[P0]]/z, [[Z0]].d, [[Z1]].d
; VBITS_GE_512-NEXT: st1w { [[Z0]].d }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret
  %a = load <8 x i64>, <8 x i64>* %ap
  %b = load <8 x i64>, <8 x i64>* %bp
  %mask = icmp eq <8 x i64> %a, %b
  %val = trunc <8 x i64> %a to <8 x i32>
  call void @llvm.masked.store.v8i32(<8 x i32> %val, <8 x i32>* %dest, i32 8, <8 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v16i32i8(<16 x i32>* %ap, <16 x i32>* %bp, <16 x i8>* %dest) #0 {
; CHECK-LABEL: masked_store_trunc_v16i32i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].s, p[[P0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: st1b { [[Z0]].s }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %a, %b
  %val = trunc <16 x i32> %a to <16 x i8>
  call void @llvm.masked.store.v16i8(<16 x i8> %val, <16 x i8>* %dest, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v16i32i16(<16 x i32>* %ap, <16 x i32>* %bp, <16 x i16>* %dest) #0 {
; CHECK-LABEL: masked_store_trunc_v16i32i16:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[Z0:z[0-9]+]].s }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[Z1:z[0-9]+]].s }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].s, p[[P0]]/z, [[Z0]].s, [[Z1]].s
; VBITS_GE_512-NEXT: st1h { [[Z0]].s }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret
  %a = load <16 x i32>, <16 x i32>* %ap
  %b = load <16 x i32>, <16 x i32>* %bp
  %mask = icmp eq <16 x i32> %a, %b
  %val = trunc <16 x i32> %a to <16 x i16>
  call void @llvm.masked.store.v16i16(<16 x i16> %val, <16 x i16>* %dest, i32 8, <16 x i1> %mask)
  ret void
}

define void @masked_store_trunc_v32i16i8(<32 x i16>* %ap, <32 x i16>* %bp, <32 x i8>* %dest) #0 {
; CHECK-LABEL: masked_store_trunc_v32i16i8:
; VBITS_GE_512: ptrue p[[P0:[0-9]+]].h, vl32
; VBITS_GE_512-NEXT: ld1h { [[Z0:z[0-9]+]].h }, p0/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[Z1:z[0-9]+]].h }, p0/z, [x1]
; VBITS_GE_512-NEXT: cmpeq p[[P1:[0-9]+]].h, p[[P0]]/z, [[Z0]].h, [[Z1]].h
; VBITS_GE_512-NEXT: st1b { [[Z0]].h }, p[[P1]], [x{{[0-9]+}}]
; VBITS_GE_512-NEXT: ret
  %a = load <32 x i16>, <32 x i16>* %ap
  %b = load <32 x i16>, <32 x i16>* %bp
  %mask = icmp eq <32 x i16> %a, %b
  %val = trunc <32 x i16> %a to <32 x i8>
  call void @llvm.masked.store.v32i8(<32 x i8> %val, <32 x i8>* %dest, i32 8, <32 x i1> %mask)
  ret void
}

declare void @llvm.masked.store.v2f16(<2 x half>, <2 x half>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2f32(<2 x float>, <2 x float>*, i32, <2 x i1>)
declare void @llvm.masked.store.v4f32(<4 x float>, <4 x float>*, i32, <4 x i1>)
declare void @llvm.masked.store.v8f32(<8 x float>, <8 x float>*, i32, <8 x i1>)
declare void @llvm.masked.store.v16f32(<16 x float>, <16 x float>*, i32, <16 x i1>)
declare void @llvm.masked.store.v32f32(<32 x float>, <32 x float>*, i32, <32 x i1>)
declare void @llvm.masked.store.v64f32(<64 x float>, <64 x float>*, i32, <64 x i1>)

declare void @llvm.masked.store.v8i8(<8 x i8>, <8 x i8>*, i32, <8 x i1>)
declare void @llvm.masked.store.v8i16(<8 x i16>, <8 x i16>*, i32, <8 x i1>)
declare void @llvm.masked.store.v8i32(<8 x i32>, <8 x i32>*, i32, <8 x i1>)
declare void @llvm.masked.store.v16i8(<16 x i8>, <16 x i8>*, i32, <16 x i1>)
declare void @llvm.masked.store.v16i16(<16 x i16>, <16 x i16>*, i32, <16 x i1>)
declare void @llvm.masked.store.v32i8(<32 x i8>, <32 x i8>*, i32, <32 x i1>)

attributes #0 = { "target-features"="+sve" }
