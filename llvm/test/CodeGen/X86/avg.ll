; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx512bw | FileCheck %s --check-prefix=AVX --check-prefix=AVX512BW

define void @avg_v4i8(<4 x i8>* %a, <4 x i8>* %b) {
; SSE2-LABEL: avg_v4i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE2-NEXT:    movd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; SSE2-NEXT:    pavgb %xmm0, %xmm1
; SSE2-NEXT:    movd %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX2-NEXT:    vmovd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; AVX2-NEXT:    vpavgb %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vmovd %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i8:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovd (%rdi), %xmm0
; AVX512BW-NEXT:    vmovd (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgb %xmm0, %xmm1, %xmm0
; AVX512BW-NEXT:    vmovd %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = load <4 x i8>, <4 x i8>* %b
  %3 = zext <4 x i8> %1 to <4 x i32>
  %4 = zext <4 x i8> %2 to <4 x i32>
  %5 = add nuw nsw <4 x i32> %3, <i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <4 x i32> %5, %4
  %7 = lshr <4 x i32> %6, <i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <4 x i32> %7 to <4 x i8>
  store <4 x i8> %8, <4 x i8>* undef, align 4
  ret void
}

define void @avg_v8i8(<8 x i8>* %a, <8 x i8>* %b) {
; SSE2-LABEL: avg_v8i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    pavgb %xmm0, %xmm1
; SSE2-NEXT:    movq %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v8i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpavgb %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v8i8:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vmovq (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgb %xmm0, %xmm1, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = load <8 x i8>, <8 x i8>* %b
  %3 = zext <8 x i8> %1 to <8 x i32>
  %4 = zext <8 x i8> %2 to <8 x i32>
  %5 = add nuw nsw <8 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <8 x i32> %5, %4
  %7 = lshr <8 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <8 x i32> %7 to <8 x i8>
  store <8 x i8> %8, <8 x i8>* undef, align 4
  ret void
}

define void @avg_v16i8(<16 x i8>* %a, <16 x i8>* %b) {
; SSE2-LABEL: avg_v16i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rsi), %xmm0
; SSE2-NEXT:    pavgb (%rdi), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rsi), %xmm0
; AVX-NEXT:    vpavgb (%rdi), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = load <16 x i8>, <16 x i8>* %b
  %3 = zext <16 x i8> %1 to <16 x i32>
  %4 = zext <16 x i8> %2 to <16 x i32>
  %5 = add nuw nsw <16 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <16 x i32> %5, %4
  %7 = lshr <16 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <16 x i32> %7 to <16 x i8>
  store <16 x i8> %8, <16 x i8>* undef, align 4
  ret void
}

define void @avg_v32i8(<32 x i8>* %a, <32 x i8>* %b) {
; AVX2-LABEL: avg_v32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rsi), %ymm0
; AVX2-NEXT:    vpavgb (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v32i8:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rsi), %ymm0
; AVX512BW-NEXT:    vpavgb (%rdi), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i8>, <32 x i8>* %a
  %2 = load <32 x i8>, <32 x i8>* %b
  %3 = zext <32 x i8> %1 to <32 x i32>
  %4 = zext <32 x i8> %2 to <32 x i32>
  %5 = add nuw nsw <32 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <32 x i32> %5, %4
  %7 = lshr <32 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <32 x i32> %7 to <32 x i8>
  store <32 x i8> %8, <32 x i8>* undef, align 4
  ret void
}

define void @avg_v64i8(<64 x i8>* %a, <64 x i8>* %b) {
; AVX512BW-LABEL: avg_v64i8:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu8 (%rsi), %zmm0
; AVX512BW-NEXT:    vpavgb (%rdi), %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu8 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <64 x i8>, <64 x i8>* %a
  %2 = load <64 x i8>, <64 x i8>* %b
  %3 = zext <64 x i8> %1 to <64 x i32>
  %4 = zext <64 x i8> %2 to <64 x i32>
  %5 = add nuw nsw <64 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <64 x i32> %5, %4
  %7 = lshr <64 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <64 x i32> %7 to <64 x i8>
  store <64 x i8> %8, <64 x i8>* undef, align 4
  ret void
}

define void @avg_v4i16(<4 x i16>* %a, <4 x i16>* %b) {
; SSE2-LABEL: avg_v4i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    pavgw %xmm0, %xmm1
; SSE2-NEXT:    movq %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpavgw %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i16:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vmovq (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgw %xmm0, %xmm1, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = load <4 x i16>, <4 x i16>* %b
  %3 = zext <4 x i16> %1 to <4 x i32>
  %4 = zext <4 x i16> %2 to <4 x i32>
  %5 = add nuw nsw <4 x i32> %3, <i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <4 x i32> %5, %4
  %7 = lshr <4 x i32> %6, <i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  store <4 x i16> %8, <4 x i16>* undef, align 4
  ret void
}

define void @avg_v8i16(<8 x i16>* %a, <8 x i16>* %b) {
; SSE2-LABEL: avg_v8i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rsi), %xmm0
; SSE2-NEXT:    pavgw (%rdi), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rsi), %xmm0
; AVX-NEXT:    vpavgw (%rdi), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = load <8 x i16>, <8 x i16>* %b
  %3 = zext <8 x i16> %1 to <8 x i32>
  %4 = zext <8 x i16> %2 to <8 x i32>
  %5 = add nuw nsw <8 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <8 x i32> %5, %4
  %7 = lshr <8 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <8 x i32> %7 to <8 x i16>
  store <8 x i16> %8, <8 x i16>* undef, align 4
  ret void
}

define void @avg_v16i16(<16 x i16>* %a, <16 x i16>* %b) {
; AVX2-LABEL: avg_v16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rsi), %ymm0
; AVX2-NEXT:    vpavgw (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v16i16:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rsi), %ymm0
; AVX512BW-NEXT:    vpavgw (%rdi), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = load <16 x i16>, <16 x i16>* %b
  %3 = zext <16 x i16> %1 to <16 x i32>
  %4 = zext <16 x i16> %2 to <16 x i32>
  %5 = add nuw nsw <16 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <16 x i32> %5, %4
  %7 = lshr <16 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <16 x i32> %7 to <16 x i16>
  store <16 x i16> %8, <16 x i16>* undef, align 4
  ret void
}

define void @avg_v32i16(<32 x i16>* %a, <32 x i16>* %b) {
; AVX512BW-LABEL: avg_v32i16:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu16 (%rsi), %zmm0
; AVX512BW-NEXT:    vpavgw (%rdi), %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu16 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i16>, <32 x i16>* %a
  %2 = load <32 x i16>, <32 x i16>* %b
  %3 = zext <32 x i16> %1 to <32 x i32>
  %4 = zext <32 x i16> %2 to <32 x i32>
  %5 = add nuw nsw <32 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %6 = add nuw nsw <32 x i32> %5, %4
  %7 = lshr <32 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <32 x i32> %7 to <32 x i16>
  store <32 x i16> %8, <32 x i16>* undef, align 4
  ret void
}

define void @avg_v4i8_2(<4 x i8>* %a, <4 x i8>* %b) {
; SSE2-LABEL: avg_v4i8_2:
; SSE2:       # BB#0:
; SSE2-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE2-NEXT:    movd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; SSE2-NEXT:    pavgb %xmm0, %xmm1
; SSE2-NEXT:    movd %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i8_2:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX2-NEXT:    vmovd {{.*#+}} xmm1 = mem[0],zero,zero,zero
; AVX2-NEXT:    vpavgb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vmovd %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i8_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovd (%rdi), %xmm0
; AVX512BW-NEXT:    vmovd (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgb %xmm1, %xmm0, %xmm0
; AVX512BW-NEXT:    vmovd %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = load <4 x i8>, <4 x i8>* %b
  %3 = zext <4 x i8> %1 to <4 x i32>
  %4 = zext <4 x i8> %2 to <4 x i32>
  %5 = add nuw nsw <4 x i32> %3, %4
  %6 = add nuw nsw <4 x i32> %5, <i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <4 x i32> %6, <i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <4 x i32> %7 to <4 x i8>
  store <4 x i8> %8, <4 x i8>* undef, align 4
  ret void
}

define void @avg_v8i8_2(<8 x i8>* %a, <8 x i8>* %b) {
; SSE2-LABEL: avg_v8i8_2:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    pavgb %xmm0, %xmm1
; SSE2-NEXT:    movq %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v8i8_2:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpavgb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v8i8_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vmovq (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgb %xmm1, %xmm0, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = load <8 x i8>, <8 x i8>* %b
  %3 = zext <8 x i8> %1 to <8 x i32>
  %4 = zext <8 x i8> %2 to <8 x i32>
  %5 = add nuw nsw <8 x i32> %3, %4
  %6 = add nuw nsw <8 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <8 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <8 x i32> %7 to <8 x i8>
  store <8 x i8> %8, <8 x i8>* undef, align 4
  ret void
}

define void @avg_v16i8_2(<16 x i8>* %a, <16 x i8>* %b) {
; SSE2-LABEL: avg_v16i8_2:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rdi), %xmm0
; SSE2-NEXT:    pavgb (%rsi), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v16i8_2:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rdi), %xmm0
; AVX-NEXT:    vpavgb (%rsi), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = load <16 x i8>, <16 x i8>* %b
  %3 = zext <16 x i8> %1 to <16 x i32>
  %4 = zext <16 x i8> %2 to <16 x i32>
  %5 = add nuw nsw <16 x i32> %3, %4
  %6 = add nuw nsw <16 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <16 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <16 x i32> %7 to <16 x i8>
  store <16 x i8> %8, <16 x i8>* undef, align 4
  ret void
}

define void @avg_v32i8_2(<32 x i8>* %a, <32 x i8>* %b) {
; AVX2-LABEL: avg_v32i8_2:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rdi), %ymm0
; AVX2-NEXT:    vpavgb (%rsi), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v32i8_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rdi), %ymm0
; AVX512BW-NEXT:    vpavgb (%rsi), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i8>, <32 x i8>* %a
  %2 = load <32 x i8>, <32 x i8>* %b
  %3 = zext <32 x i8> %1 to <32 x i32>
  %4 = zext <32 x i8> %2 to <32 x i32>
  %5 = add nuw nsw <32 x i32> %3, %4
  %6 = add nuw nsw <32 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <32 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <32 x i32> %7 to <32 x i8>
  store <32 x i8> %8, <32 x i8>* undef, align 4
  ret void
}

define void @avg_v64i8_2(<64 x i8>* %a, <64 x i8>* %b) {
; AVX512BW-LABEL: avg_v64i8_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu8 (%rsi), %zmm0
; AVX512BW-NEXT:    vpavgb %zmm0, %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu8 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <64 x i8>, <64 x i8>* %a
  %2 = load <64 x i8>, <64 x i8>* %b
  %3 = zext <64 x i8> %1 to <64 x i32>
  %4 = zext <64 x i8> %2 to <64 x i32>
  %5 = add nuw nsw <64 x i32> %4, %4
  %6 = add nuw nsw <64 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <64 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <64 x i32> %7 to <64 x i8>
  store <64 x i8> %8, <64 x i8>* undef, align 4
  ret void
}


define void @avg_v4i16_2(<4 x i16>* %a, <4 x i16>* %b) {
; SSE2-LABEL: avg_v4i16_2:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    pavgw %xmm0, %xmm1
; SSE2-NEXT:    movq %xmm1, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i16_2:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; AVX2-NEXT:    vpavgw %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i16_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vmovq (%rsi), %xmm1
; AVX512BW-NEXT:    vpavgw %xmm1, %xmm0, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = load <4 x i16>, <4 x i16>* %b
  %3 = zext <4 x i16> %1 to <4 x i32>
  %4 = zext <4 x i16> %2 to <4 x i32>
  %5 = add nuw nsw <4 x i32> %3, %4
  %6 = add nuw nsw <4 x i32> %5, <i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <4 x i32> %6, <i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  store <4 x i16> %8, <4 x i16>* undef, align 4
  ret void
}

define void @avg_v8i16_2(<8 x i16>* %a, <8 x i16>* %b) {
; SSE2-LABEL: avg_v8i16_2:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rdi), %xmm0
; SSE2-NEXT:    pavgw (%rsi), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v8i16_2:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rdi), %xmm0
; AVX-NEXT:    vpavgw (%rsi), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = load <8 x i16>, <8 x i16>* %b
  %3 = zext <8 x i16> %1 to <8 x i32>
  %4 = zext <8 x i16> %2 to <8 x i32>
  %5 = add nuw nsw <8 x i32> %3, %4
  %6 = add nuw nsw <8 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <8 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <8 x i32> %7 to <8 x i16>
  store <8 x i16> %8, <8 x i16>* undef, align 4
  ret void
}

define void @avg_v16i16_2(<16 x i16>* %a, <16 x i16>* %b) {
; AVX2-LABEL: avg_v16i16_2:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rdi), %ymm0
; AVX2-NEXT:    vpavgw (%rsi), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v16i16_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rdi), %ymm0
; AVX512BW-NEXT:    vpavgw (%rsi), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = load <16 x i16>, <16 x i16>* %b
  %3 = zext <16 x i16> %1 to <16 x i32>
  %4 = zext <16 x i16> %2 to <16 x i32>
  %5 = add nuw nsw <16 x i32> %3, %4
  %6 = add nuw nsw <16 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <16 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <16 x i32> %7 to <16 x i16>
  store <16 x i16> %8, <16 x i16>* undef, align 4
  ret void
}

define void @avg_v32i16_2(<32 x i16>* %a, <32 x i16>* %b) {
; AVX512BW-LABEL: avg_v32i16_2:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu16 (%rdi), %zmm0
; AVX512BW-NEXT:    vpavgw (%rsi), %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu16 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i16>, <32 x i16>* %a
  %2 = load <32 x i16>, <32 x i16>* %b
  %3 = zext <32 x i16> %1 to <32 x i32>
  %4 = zext <32 x i16> %2 to <32 x i32>
  %5 = add nuw nsw <32 x i32> %3, %4
  %6 = add nuw nsw <32 x i32> %5, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %7 = lshr <32 x i32> %6, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %8 = trunc <32 x i32> %7 to <32 x i16>
  store <32 x i16> %8, <32 x i16>* undef, align 4
  ret void
}

define void @avg_v4i8_const(<4 x i8>* %a) {
; SSE2-LABEL: avg_v4i8_const:
; SSE2:       # BB#0:
; SSE2-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE2-NEXT:    pavgb {{.*}}(%rip), %xmm0
; SSE2-NEXT:    movd %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i8_const:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; AVX2-NEXT:    vpavgb {{.*}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vmovd %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i8_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovd (%rdi), %xmm0
; AVX512BW-NEXT:    vpavgb {{.*}}(%rip), %xmm0, %xmm0
; AVX512BW-NEXT:    vmovd %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = zext <4 x i8> %1 to <4 x i32>
  %3 = add nuw nsw <4 x i32> %2, <i32 1, i32 2, i32 3, i32 4>
  %4 = lshr <4 x i32> %3, <i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <4 x i32> %4 to <4 x i8>
  store <4 x i8> %5, <4 x i8>* undef, align 4
  ret void
}

define void @avg_v8i8_const(<8 x i8>* %a) {
; SSE2-LABEL: avg_v8i8_const:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    pavgb {{.*}}(%rip), %xmm0
; SSE2-NEXT:    movq %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v8i8_const:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vpavgb {{.*}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v8i8_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vpavgb {{.*}}(%rip), %xmm0, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = zext <8 x i8> %1 to <8 x i32>
  %3 = add nuw nsw <8 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <8 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <8 x i32> %4 to <8 x i8>
  store <8 x i8> %5, <8 x i8>* undef, align 4
  ret void
}

define void @avg_v16i8_const(<16 x i8>* %a) {
; SSE2-LABEL: avg_v16i8_const:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rdi), %xmm0
; SSE2-NEXT:    pavgb {{.*}}(%rip), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v16i8_const:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rdi), %xmm0
; AVX-NEXT:    vpavgb {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = zext <16 x i8> %1 to <16 x i32>
  %3 = add nuw nsw <16 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <16 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <16 x i32> %4 to <16 x i8>
  store <16 x i8> %5, <16 x i8>* undef, align 4
  ret void
}

define void @avg_v32i8_const(<32 x i8>* %a) {
; AVX2-LABEL: avg_v32i8_const:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rdi), %ymm0
; AVX2-NEXT:    vpavgb {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v32i8_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rdi), %ymm0
; AVX512BW-NEXT:    vpavgb {{.*}}(%rip), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i8>, <32 x i8>* %a
  %2 = zext <32 x i8> %1 to <32 x i32>
  %3 = add nuw nsw <32 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <32 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <32 x i32> %4 to <32 x i8>
  store <32 x i8> %5, <32 x i8>* undef, align 4
  ret void
}

define void @avg_v64i8_const(<64 x i8>* %a) {
; AVX512BW-LABEL: avg_v64i8_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu8 (%rdi), %zmm0
; AVX512BW-NEXT:    vpavgb {{.*}}(%rip), %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu8 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <64 x i8>, <64 x i8>* %a
  %2 = zext <64 x i8> %1 to <64 x i32>
  %3 = add nuw nsw <64 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <64 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <64 x i32> %4 to <64 x i8>
  store <64 x i8> %5, <64 x i8>* undef, align 4
  ret void
}

define void @avg_v4i16_const(<4 x i16>* %a) {
; SSE2-LABEL: avg_v4i16_const:
; SSE2:       # BB#0:
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    pavgw {{.*}}(%rip), %xmm0
; SSE2-NEXT:    movq %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX2-LABEL: avg_v4i16_const:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; AVX2-NEXT:    vpavgw {{.*}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vmovq %xmm0, (%rax)
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v4i16_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovq (%rdi), %xmm0
; AVX512BW-NEXT:    vpavgw {{.*}}(%rip), %xmm0, %xmm0
; AVX512BW-NEXT:    vmovq %xmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = zext <4 x i16> %1 to <4 x i32>
  %3 = add nuw nsw <4 x i32> %2, <i32 1, i32 2, i32 3, i32 4>
  %4 = lshr <4 x i32> %3, <i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <4 x i32> %4 to <4 x i16>
  store <4 x i16> %5, <4 x i16>* undef, align 4
  ret void
}

define void @avg_v8i16_const(<8 x i16>* %a) {
; SSE2-LABEL: avg_v8i16_const:
; SSE2:       # BB#0:
; SSE2-NEXT:    movdqa (%rdi), %xmm0
; SSE2-NEXT:    pavgw {{.*}}(%rip), %xmm0
; SSE2-NEXT:    movdqu %xmm0, (%rax)
; SSE2-NEXT:    retq
;
; AVX-LABEL: avg_v8i16_const:
; AVX:       # BB#0:
; AVX-NEXT:    vmovdqa (%rdi), %xmm0
; AVX-NEXT:    vpavgw {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqu %xmm0, (%rax)
; AVX-NEXT:    retq
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = zext <8 x i16> %1 to <8 x i32>
  %3 = add nuw nsw <8 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <8 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <8 x i32> %4 to <8 x i16>
  store <8 x i16> %5, <8 x i16>* undef, align 4
  ret void
}

define void @avg_v16i16_const(<16 x i16>* %a) {
; AVX2-LABEL: avg_v16i16_const:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa (%rdi), %ymm0
; AVX2-NEXT:    vpavgw {{.*}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:    vmovdqu %ymm0, (%rax)
; AVX2-NEXT:    vzeroupper
; AVX2-NEXT:    retq
;
; AVX512BW-LABEL: avg_v16i16_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqa (%rdi), %ymm0
; AVX512BW-NEXT:    vpavgw {{.*}}(%rip), %ymm0, %ymm0
; AVX512BW-NEXT:    vmovdqu %ymm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = zext <16 x i16> %1 to <16 x i32>
  %3 = add nuw nsw <16 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <16 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <16 x i32> %4 to <16 x i16>
  store <16 x i16> %5, <16 x i16>* undef, align 4
  ret void
}

define void @avg_v32i16_const(<32 x i16>* %a) {
; AVX512BW-LABEL: avg_v32i16_const:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vmovdqu16 (%rdi), %zmm0
; AVX512BW-NEXT:    vpavgw {{.*}}(%rip), %zmm0, %zmm0
; AVX512BW-NEXT:    vmovdqu16 %zmm0, (%rax)
; AVX512BW-NEXT:    retq
  %1 = load <32 x i16>, <32 x i16>* %a
  %2 = zext <32 x i16> %1 to <32 x i32>
  %3 = add nuw nsw <32 x i32> %2, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %4 = lshr <32 x i32> %3, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %5 = trunc <32 x i32> %4 to <32 x i16>
  store <32 x i16> %5, <32 x i16>* undef, align 4
  ret void
}
