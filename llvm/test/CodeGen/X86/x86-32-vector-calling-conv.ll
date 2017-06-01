; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+avx512f | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=i686-pc-linux -mattr=+avx512f | FileCheck %s --check-prefix=LINUX

; CHECK-LABEL: test_sse:
; DARWIN: vpaddd  %xmm3, %xmm2, %xmm2
; DARWIN: vpaddd  %xmm2, %xmm1, %xmm1
; DARWIN: vpaddd  %xmm1, %xmm0, %xmm0
; LINUX:  vpaddd  {{[0-9]+}}(%e{{s|b}}p), %xmm2, %xmm2
; LINUX:  vpaddd  %xmm2, %xmm1, %xmm1
; LINUX:  vpaddd  %xmm1, %xmm0, %xmm0
define <4 x i32> @test_sse(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, <4 x i32> %d) nounwind {
  %r0 = add <4 x i32> %a, %b
  %r1 = add <4 x i32> %c, %d
  %ret = add <4 x i32> %r0, %r1
  ret <4 x i32> %ret
}

; CHECK-LABEL: test_avx:
; DARWIN: vpaddd  %ymm3, %ymm2, %ymm2
; DARWIN: vpaddd  %ymm2, %ymm1, %ymm1
; DARWIN: vpaddd  %ymm1, %ymm0, %ymm0
; LINUX:  vpaddd  {{[0-9]+}}(%e{{s|b}}p), %ymm2, %ymm2
; LINUX:  vpaddd  %ymm2, %ymm1, %ymm1
; LINUX:  vpaddd  %ymm1, %ymm0, %ymm0
define <8 x i32> @test_avx(<8 x i32> %a, <8 x i32> %b, <8 x i32> %c, <8 x i32> %d) nounwind {
  %r0 = add <8 x i32> %a, %b
  %r1 = add <8 x i32> %c, %d
  %ret = add <8 x i32> %r0, %r1
  ret <8 x i32> %ret
}

; CHECK-LABEL: test_avx512:
; DARWIN: vpaddd  %zmm3, %zmm2, %zmm2
; DARWIN: vpaddd  %zmm2, %zmm1, %zmm1
; DARWIN: vpaddd  %zmm1, %zmm0, %zmm0
; LINUX:  vpaddd  {{[0-9]+}}(%e{{s|b}}p), %zmm2, %zmm2
; LINUX:  vpaddd  %zmm2, %zmm1, %zmm1
; LINUX:  vpaddd  %zmm1, %zmm0, %zmm0
define <16 x i32> @test_avx512(<16 x i32> %a, <16 x i32> %b, <16 x i32> %c, <16 x i32> %d) nounwind {
  %r0 = add <16 x i32> %a, %b
  %r1 = add <16 x i32> %c, %d
  %ret = add <16 x i32> %r0, %r1
  ret <16 x i32> %ret
}
