; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+vaes,+avx512f -show-mc-encoding | FileCheck %s --check-prefix=VAES_AVX512

define <8 x i64> @test_x86_aesni_aesenc_512(<8 x i64> %a0, <8 x i64> %a1) {
; VAES_AVX512-LABEL: test_x86_aesni_aesenc_512:
; VAES_AVX512:       # %bb.0:
; VAES_AVX512-NEXT:    vaesenc %zmm1, %zmm0, %zmm0 # encoding: [0x62,0xf2,0x7d,0x48,0xdc,0xc1]
; VAES_AVX512-NEXT:    retq # encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.aesni.aesenc.512(<8 x i64> %a0, <8 x i64> %a1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.aesni.aesenc.512(<8 x i64>, <8 x i64>) nounwind readnone

define <8 x i64> @test_x86_aesni_aesenclast_512(<8 x i64> %a0, <8 x i64> %a1) {
; VAES_AVX512-LABEL: test_x86_aesni_aesenclast_512:
; VAES_AVX512:       # %bb.0:
; VAES_AVX512-NEXT:    vaesenclast %zmm1, %zmm0, %zmm0 # encoding: [0x62,0xf2,0x7d,0x48,0xdd,0xc1]
; VAES_AVX512-NEXT:    retq # encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.aesni.aesenclast.512(<8 x i64> %a0, <8 x i64> %a1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.aesni.aesenclast.512(<8 x i64>, <8 x i64>) nounwind readnone

define <8 x i64> @test_x86_aesni_aesdec_512(<8 x i64> %a0, <8 x i64> %a1) {
; VAES_AVX512-LABEL: test_x86_aesni_aesdec_512:
; VAES_AVX512:       # %bb.0:
; VAES_AVX512-NEXT:    vaesdec %zmm1, %zmm0, %zmm0 # encoding: [0x62,0xf2,0x7d,0x48,0xde,0xc1]
; VAES_AVX512-NEXT:    retq # encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.aesni.aesdec.512(<8 x i64> %a0, <8 x i64> %a1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.aesni.aesdec.512(<8 x i64>, <8 x i64>) nounwind readnone

define <8 x i64> @test_x86_aesni_aesdeclast_512(<8 x i64> %a0, <8 x i64> %a1) {
; VAES_AVX512-LABEL: test_x86_aesni_aesdeclast_512:
; VAES_AVX512:       # %bb.0:
; VAES_AVX512-NEXT:    vaesdeclast %zmm1, %zmm0, %zmm0 # encoding: [0x62,0xf2,0x7d,0x48,0xdf,0xc1]
; VAES_AVX512-NEXT:    retq # encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.aesni.aesdeclast.512(<8 x i64> %a0, <8 x i64> %a1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.aesni.aesdeclast.512(<8 x i64>, <8 x i64>) nounwind readnone

