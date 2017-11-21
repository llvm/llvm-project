; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+vaes,+avx512f,+avx512vl -show-mc-encoding | FileCheck %s --check-prefix=VAES_AVX512VL

define <2 x i64> @test_x86_aesni_aesenc(<2 x i64> %a0, <2 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesenc:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesenc %xmm1, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0xdc,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @test_x86_aesni_aesenc_256(<4 x i64> %a0, <4 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesenc_256:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesenc %ymm1, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x7d,0xdc,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <4 x i64> @llvm.x86.aesni.aesenc.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.aesni.aesenc.256(<4 x i64>, <4 x i64>) nounwind readnone

define <2 x i64> @test_x86_aesni_aesenclast(<2 x i64> %a0, <2 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesenclast:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesenclast %xmm1, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0xdd,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @test_x86_aesni_aesenclast_256(<4 x i64> %a0, <4 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesenclast_256:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesenclast %ymm1, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x7d,0xdd,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <4 x i64> @llvm.x86.aesni.aesenclast.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.aesni.aesenclast.256(<4 x i64>, <4 x i64>) nounwind readnone

define <2 x i64> @test_x86_aesni_aesdec(<2 x i64> %a0, <2 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesdec:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesdec %xmm1, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0xde,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @test_x86_aesni_aesdec_256(<4 x i64> %a0, <4 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesdec_256:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesdec %ymm1, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x7d,0xde,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <4 x i64> @llvm.x86.aesni.aesdec.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.aesni.aesdec.256(<4 x i64>, <4 x i64>) nounwind readnone

define <2 x i64> @test_x86_aesni_aesdeclast(<2 x i64> %a0, <2 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesdeclast:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesdeclast %xmm1, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0xdf,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i64> @test_x86_aesni_aesdeclast_256(<4 x i64> %a0, <4 x i64> %a1) {
; VAES_AVX512VL-LABEL: test_x86_aesni_aesdeclast_256:
; VAES_AVX512VL:       # BB#0:
; VAES_AVX512VL-NEXT:    vaesdeclast %ymm1, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc4,0xe2,0x7d,0xdf,0xc1]
; VAES_AVX512VL-NEXT:    retq # encoding: [0xc3]
  %res = call <4 x i64> @llvm.x86.aesni.aesdeclast.256(<4 x i64> %a0, <4 x i64> %a1)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.aesni.aesdeclast.256(<4 x i64>, <4 x i64>) nounwind readnone

