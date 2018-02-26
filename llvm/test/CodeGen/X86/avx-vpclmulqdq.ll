; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=avx,vpclmulqdq -show-mc-encoding | FileCheck %s --check-prefix=AVX_VPCLMULQDQ

; Check for vpclmulqdq
define <4 x i64> @test_x86_pclmulqdq(<4 x i64> %a0, <4 x i64> %a1) {
; AVX_VPCLMULQDQ-LABEL: test_x86_pclmulqdq:
; AVX_VPCLMULQDQ:       # %bb.0:
; AVX_VPCLMULQDQ-NEXT:    vpclmulqdq $17, %ymm1, %ymm0, %ymm0 # encoding: [0xc4,0xe3,0x7d,0x44,0xc1,0x11]
; AVX_VPCLMULQDQ-NEXT:    retl # encoding: [0xc3]
  %res = call <4 x i64> @llvm.x86.pclmulqdq.256(<4 x i64> %a0, <4 x i64> %a1, i8 17)
  ret <4 x i64> %res
}
declare <4 x i64> @llvm.x86.pclmulqdq.256(<4 x i64>, <4 x i64>, i8) nounwind readnone

