; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+vpclmulqdq -show-mc-encoding | FileCheck %s --check-prefix=AVX512_VPCLMULQDQ

define <8 x i64> @test_x86_pclmulqdq(<8 x i64> %a0, <8 x i64> %a1) {
; AVX512_VPCLMULQDQ-LABEL: test_x86_pclmulqdq:
; AVX512_VPCLMULQDQ:       # BB#0:
; AVX512_VPCLMULQDQ-NEXT:    vpclmulqdq $1, %zmm1, %zmm0, %zmm0 # encoding: [0x62,0xf3,0x7d,0x48,0x44,0xc1,0x01]
; AVX512_VPCLMULQDQ-NEXT:    retq # encoding: [0xc3]
  %res = call <8 x i64> @llvm.x86.pclmulqdq.512(<8 x i64> %a0, <8 x i64> %a1, i8 1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.pclmulqdq.512(<8 x i64>, <8 x i64>, i8) nounwind readnone
