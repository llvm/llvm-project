; avx10.x-512 is just avx10.x -- 512 is kept for compatibility purposes.

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx10.1-512 2>&1 | FileCheck --check-prefixes=CHECK-AVX10_1 %s

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx10.2-512 2>&1 | FileCheck --check-prefixes=CHECK-AVX10_2 %s

; CHECK-AVX10_1-NOT: is not recognizable
; CHECK-AVX10_2-NOT: is not recognizable

define <32 x bfloat> @foo_avx10.1(<16 x float> %a, <16 x float> %b) {
  %ret = call <32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a, <16 x float> %b)
  ret <32 x bfloat> %ret
}

define <8 x i32> @foo_avx10.2(<8 x double> %f) {
  %x = call  <8 x i32> @llvm.fptosi.sat.v8i32.v8f64(<8 x double> %f)
  ret <8 x i32> %x
}

