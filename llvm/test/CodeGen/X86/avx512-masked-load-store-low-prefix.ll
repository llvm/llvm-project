; RUN: llc -mtriple=x86_64-apple-darwin -mattr=+avx512f,+avx512bw,+avx512vl < %s | FileCheck %s

define void @mload_mstore_v5i64_low4(ptr %src, ptr %dst) {
; CHECK-LABEL: mload_mstore_v5i64_low4:
; CHECK:       ## %bb.0:
; CHECK-NOT:     {%k
; CHECK:         vmovups	(%rdi), %ymm0
; CHECK-NOT:     {%k
; CHECK:         vmovups	%ymm0, (%rsi)
; CHECK-NOT:     {%k
; CHECK:         retq
  %val = tail call <5 x i64> @llvm.masked.load.v5i64.p0(ptr %src, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>, <5 x i64> poison)
  tail call void @llvm.masked.store.v5i64.p0(<5 x i64> %val, ptr %dst, <5 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @mload_mstore_v9i32_low8(ptr %src, ptr %dst) {
; CHECK-LABEL: mload_mstore_v9i32_low8:
; CHECK:       ## %bb.0:
; CHECK-NOT:     {%k
; CHECK:         vmovups	(%rdi), %ymm0
; CHECK-NOT:     {%k
; CHECK:         vmovups	%ymm0, (%rsi)
; CHECK-NOT:     {%k
; CHECK:         retq
  %val = tail call <9 x i32> @llvm.masked.load.v9i32.p0(ptr %src, <9 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <9 x i32> poison)
  tail call void @llvm.masked.store.v9i32.p0(<9 x i32> %val, ptr %dst, <9 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @mload_mstore_v17i8_low16(ptr %src, ptr %dst) {
; CHECK-LABEL: mload_mstore_v17i8_low16:
; CHECK:       ## %bb.0:
; CHECK-NOT:     {%k
; CHECK:         vmovups	(%rdi), %xmm0
; CHECK-NOT:     {%k
; CHECK:         vmovups	%xmm0, (%rsi)
; CHECK-NOT:     {%k
; CHECK:         retq
  %val = tail call <17 x i8> @llvm.masked.load.v17i8.p0(ptr %src, <17 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <17 x i8> poison)
  tail call void @llvm.masked.store.v17i8.p0(<17 x i8> %val, ptr %dst, <17 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @mload_mstore_v33i8_low32(ptr %src, ptr %dst) {
; CHECK-LABEL: mload_mstore_v33i8_low32:
; CHECK:       ## %bb.0:
; CHECK-NOT:     {%k
; CHECK:         vmovups	(%rdi), %ymm0
; CHECK-NOT:     {%k
; CHECK:         vmovups	%ymm0, (%rsi)
; CHECK-NOT:     {%k
; CHECK:         retq
  %val = tail call <33 x i8> @llvm.masked.load.v33i8.p0(ptr %src, <33 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>, <33 x i8> poison)
  tail call void @llvm.masked.store.v33i8.p0(<33 x i8> %val, ptr %dst, <33 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @mstore_v9i32_gap_mask(ptr %dst, <9 x i32> %val) {
; CHECK-LABEL: mstore_v9i32_gap_mask:
; CHECK:       ## %bb.0:
; CHECK:         {%k
; CHECK:         retq
  tail call void @llvm.masked.store.v9i32.p0(<9 x i32> %val, ptr %dst, <9 x i1> <i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

define void @mstore_v9i32_high_mask(ptr %dst, <9 x i32> %val) {
; CHECK-LABEL: mstore_v9i32_high_mask:
; CHECK:       ## %bb.0:
; CHECK:         {%k
; CHECK:         retq
  tail call void @llvm.masked.store.v9i32.p0(<9 x i32> %val, ptr %dst, <9 x i1> <i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false>)
  ret void
}

declare <5 x i64> @llvm.masked.load.v5i64.p0(ptr, <5 x i1>, <5 x i64>)
declare void @llvm.masked.store.v5i64.p0(<5 x i64>, ptr, <5 x i1>)
declare <9 x i32> @llvm.masked.load.v9i32.p0(ptr, <9 x i1>, <9 x i32>)
declare void @llvm.masked.store.v9i32.p0(<9 x i32>, ptr, <9 x i1>)
declare <17 x i8> @llvm.masked.load.v17i8.p0(ptr, <17 x i1>, <17 x i8>)
declare void @llvm.masked.store.v17i8.p0(<17 x i8>, ptr, <17 x i1>)
declare <33 x i8> @llvm.masked.load.v33i8.p0(ptr, <33 x i1>, <33 x i8>)
declare void @llvm.masked.store.v33i8.p0(<33 x i8>, ptr, <33 x i1>)
