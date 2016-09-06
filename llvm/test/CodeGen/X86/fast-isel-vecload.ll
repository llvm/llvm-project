; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -mattr=+sse2 < %s | FileCheck %s --check-prefix=SSE --check-prefix=ALL
; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s --check-prefix=AVX --check-prefix=ALL
; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -mattr=+avx512f < %s | FileCheck %s --check-prefix=KNL

; Verify that fast-isel knows how to select aligned/unaligned vector loads.
; Also verify that the selected load instruction is in the correct domain.

define <16 x i8> @test_v16i8(<16 x i8>* %V) {
; ALL-LABEL: test_v16i8:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <16 x i8>, <16 x i8>* %V, align 16
  ret <16 x i8> %0
}

define <8 x i16> @test_v8i16(<8 x i16>* %V) {
; ALL-LABEL: test_v8i16:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <8 x i16>, <8 x i16>* %V, align 16
  ret <8 x i16> %0
}

define <4 x i32> @test_v4i32(<4 x i32>* %V) {
; ALL-LABEL: test_v4i32:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x i32>, <4 x i32>* %V, align 16
  ret <4 x i32> %0
}

define <2 x i64> @test_v2i64(<2 x i64>* %V) {
; ALL-LABEL: test_v2i64:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x i64>, <2 x i64>* %V, align 16
  ret <2 x i64> %0
}

define <16 x i8> @test_v16i8_unaligned(<16 x i8>* %V) {
; ALL-LABEL: test_v16i8_unaligned:
; SSE: movdqu  (%rdi), %xmm0
; AVX: vmovdqu  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <16 x i8>, <16 x i8>* %V, align 4
  ret <16 x i8> %0
}

define <8 x i16> @test_v8i16_unaligned(<8 x i16>* %V) {
; ALL-LABEL: test_v8i16_unaligned:
; SSE: movdqu  (%rdi), %xmm0
; AVX: vmovdqu  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <8 x i16>, <8 x i16>* %V, align 4
  ret <8 x i16> %0
}

define <4 x i32> @test_v4i32_unaligned(<4 x i32>* %V) {
; ALL-LABEL: test_v4i32_unaligned:
; SSE: movdqu  (%rdi), %xmm0
; AVX: vmovdqu  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x i32>, <4 x i32>* %V, align 4
  ret <4 x i32> %0
}

define <2 x i64> @test_v2i64_unaligned(<2 x i64>* %V) {
; ALL-LABEL: test_v2i64_unaligned:
; SSE: movdqu  (%rdi), %xmm0
; AVX: vmovdqu  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x i64>, <2 x i64>* %V, align 4
  ret <2 x i64> %0
}

define <4 x float> @test_v4f32(<4 x float>* %V) {
; ALL-LABEL: test_v4f32:
; SSE: movaps  (%rdi), %xmm0
; AVX: vmovaps  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x float>, <4 x float>* %V, align 16
  ret <4 x float> %0
}

define <2 x double> @test_v2f64(<2 x double>* %V) {
; ALL-LABEL: test_v2f64:
; SSE: movapd  (%rdi), %xmm0
; AVX: vmovapd  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x double>, <2 x double>* %V, align 16
  ret <2 x double> %0
}

define <4 x float> @test_v4f32_unaligned(<4 x float>* %V) {
; ALL-LABEL: test_v4f32_unaligned:
; SSE: movups  (%rdi), %xmm0
; AVX: vmovups  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x float>, <4 x float>* %V, align 4
  ret <4 x float> %0
}

define <2 x double> @test_v2f64_unaligned(<2 x double>* %V) {
; ALL-LABEL: test_v2f64_unaligned:
; SSE: movupd  (%rdi), %xmm0
; AVX: vmovupd  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x double>, <2 x double>* %V, align 4
  ret <2 x double> %0
}

define <16 x i8> @test_v16i8_abi_alignment(<16 x i8>* %V) {
; ALL-LABEL: test_v16i8_abi_alignment:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <16 x i8>, <16 x i8>* %V
  ret <16 x i8> %0
}

define <8 x i16> @test_v8i16_abi_alignment(<8 x i16>* %V) {
; ALL-LABEL: test_v8i16_abi_alignment:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <8 x i16>, <8 x i16>* %V
  ret <8 x i16> %0
}

define <4 x i32> @test_v4i32_abi_alignment(<4 x i32>* %V) {
; ALL-LABEL: test_v4i32_abi_alignment:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x i32>, <4 x i32>* %V
  ret <4 x i32> %0
}

define <2 x i64> @test_v2i64_abi_alignment(<2 x i64>* %V) {
; ALL-LABEL: test_v2i64_abi_alignment:
; SSE: movdqa  (%rdi), %xmm0
; AVX: vmovdqa  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x i64>, <2 x i64>* %V
  ret <2 x i64> %0
}

define <4 x float> @test_v4f32_abi_alignment(<4 x float>* %V) {
; ALL-LABEL: test_v4f32_abi_alignment:
; SSE: movaps  (%rdi), %xmm0
; AVX: vmovaps  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <4 x float>, <4 x float>* %V
  ret <4 x float> %0
}

define <2 x double> @test_v2f64_abi_alignment(<2 x double>* %V) {
; ALL-LABEL: test_v2f64_abi_alignment:
; SSE: movapd  (%rdi), %xmm0
; AVX: vmovapd  (%rdi), %xmm0
; ALL-NEXT: retq
entry:
  %0 = load <2 x double>, <2 x double>* %V
  ret <2 x double> %0
}

define <8 x i64> @test_v8i64_alignment(<8 x i64>* %V) {
; KNL-LABEL: test_v8i64_alignment:
; KNL:       # BB#0: # %entry
; KNL-NEXT:    vmovdqa64 (%rdi), %zmm0
; KNL-NEXT:    retq
entry:
  %0 = load <8 x i64>, <8 x i64>* %V, align 64
  ret <8 x i64> %0
}

define <8 x i64> @test_v8i64(<8 x i64>* %V) {
; KNL-LABEL: test_v8i64:
; KNL:       # BB#0: # %entry
; KNL-NEXT:    vmovdqu64 (%rdi), %zmm0
; KNL-NEXT:    retq
entry:
  %0 = load <8 x i64>, <8 x i64>* %V, align 4
  ret <8 x i64> %0
}
