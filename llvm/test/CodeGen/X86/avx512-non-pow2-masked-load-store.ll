; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vl | FileCheck %s

; These cases should avoid AVX-512 mask materialization and masked memory
; operations for a constant true-prefix mask. They are equivalent to an
; unmasked power-of-two-width load/store over the active prefix:
; v5i64 -> v4i64, v9i32 -> v8i32, v17i8 -> v16i8, v33i8 -> v32i8.

define void @load_store_masked_v5i64(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_masked_v5i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups (%rdi), %ymm0
; CHECK-NEXT:    vmovups %ymm0, (%rsi)
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>, <5 x i64> poison)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_masked_v9i32(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_masked_v9i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups (%rdi), %ymm0
; CHECK-NEXT:    vmovups %ymm0, (%rsi)
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <9 x i32> @llvm.masked.load.v9i32.p0(ptr %src, i32 1, <9 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <9 x i32> poison)
  call void @llvm.masked.store.v9i32.p0(<9 x i32> %val, ptr %dst, i32 1, <9 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_masked_v17i8(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_masked_v17i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups (%rdi), %xmm0
; CHECK-NEXT:    vmovups %xmm0, (%rsi)
; CHECK-NEXT:    retq
  %val = call <17 x i8> @llvm.masked.load.v17i8.p0(ptr %src, i32 1, <17 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <17 x i8> poison)
  call void @llvm.masked.store.v17i8.p0(<17 x i8> %val, ptr %dst, i32 1, <17 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_masked_v33i8(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_masked_v33i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups (%rdi), %ymm0
; CHECK-NEXT:    vmovups %ymm0, (%rsi)
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <33 x i8> @llvm.masked.load.v33i8.p0(ptr %src, i32 1, <33 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <33 x i8> poison)
  call void @llvm.masked.store.v33i8.p0(<33 x i8> %val, ptr %dst, i32 1, <33 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_non_prefix_mask_v5i64(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_non_prefix_mask_v5i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movb $13, %al
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    vmovdqu64 %zmm0, (%rsi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, i32 1, <5 x i1> <i1 true, i1 false, i1 true, i1 true, i1 false>, <5 x i64> poison)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst, i32 1, <5 x i1> <i1 true, i1 false, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_prefix3_mask_v5i64(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_prefix3_mask_v5i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movb $7, %al
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    vmovdqu64 %zmm0, (%rsi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false>, <5 x i64> poison)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false>)
  ret void
}

define void @load_store_passthru_zero_v5i64(ptr %src, ptr %dst) {
; CHECK-LABEL: load_store_passthru_zero_v5i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movb $15, %al
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    vmovdqu64 %zmm0, (%rsi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>, <5 x i64> zeroinitializer)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @load_store_two_uses_v5i64(ptr %src, ptr %dst0, ptr %dst1) {
; CHECK-LABEL: load_store_two_uses_v5i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movb $15, %al
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vmovdqu64 (%rdi), %zmm0 {%k1} {z}
; CHECK-NEXT:    vmovdqu64 %zmm0, (%rsi) {%k1}
; CHECK-NEXT:    vmovdqu64 %zmm0, (%rdx) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %val = call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>, <5 x i64> poison)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst0, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>)
  call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst1, i32 1, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

declare <5 x i64> @llvm.masked.load.v5i64.p0(ptr, i32 immarg, <5 x i1>, <5 x i64>)
declare void @llvm.masked.store.v5i64.p0(<5 x i64>, ptr, i32 immarg, <5 x i1>)
declare <9 x i32> @llvm.masked.load.v9i32.p0(ptr, i32 immarg, <9 x i1>, <9 x i32>)
declare void @llvm.masked.store.v9i32.p0(<9 x i32>, ptr, i32 immarg, <9 x i1>)
declare <17 x i8> @llvm.masked.load.v17i8.p0(ptr, i32 immarg, <17 x i1>, <17 x i8>)
declare void @llvm.masked.store.v17i8.p0(<17 x i8>, ptr, i32 immarg, <17 x i1>)
declare <33 x i8> @llvm.masked.load.v33i8.p0(ptr, i32 immarg, <33 x i1>, <33 x i8>)
declare void @llvm.masked.store.v33i8.p0(<33 x i8>, ptr, i32 immarg, <33 x i1>)
