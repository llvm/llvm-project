; RUN: llc < %s -mtriple=x86_64-- -mattr=+avxifma | FileCheck %s --check-prefixes=X64,AVX
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx512ifma | FileCheck %s --check-prefixes=X64,AVX512,AVX512-NOVL
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx512ifma,+avx512vl | FileCheck %s --check-prefixes=X64,AVX512,AVX512VL

; 67108863 == (1 << 26) - 1
; 4503599627370496 == (1 << 52)
; 4503599627370495 == (1 << 52) - 1

define <8 x i64> @test_512_combine(<8 x i64> %x, <8 x i64> %y, <8 x i64> %z) {
  %x_masked = and <8 x i64> %x, splat (i64 67108863)
  %y_masked = and <8 x i64> %y, splat (i64 67108863)
  %mul = mul nuw nsw <8 x i64> %x_masked, %y_masked
  %res = add nuw nsw <8 x i64> %mul, %z
  ret <8 x i64> %res
}

define <8 x i64> @test_512_combine_v2(<8 x i64> %x, <8 x i64> %y, <8 x i64> %z) {
  %x_masked = and <8 x i64> %x, splat (i64 1125899906842623) ; (1 << 50) - 1
  %y_masked = and <8 x i64> %y, splat (i64 3)
  %mul = mul nuw nsw <8 x i64> %x_masked, %y_masked
  %res = add nuw nsw <8 x i64> %mul, %z
  ret <8 x i64> %res
}

define <8 x i64> @test_512_no_combine(<8 x i64> %x, <8 x i64> %y, <8 x i64> %z) {
  %x_masked = and <8 x i64> %x, splat (i64 4503599627370495)
  %y_masked = and <8 x i64> %y, splat (i64 4503599627370495)
  %mul = mul nuw nsw <8 x i64> %x_masked, %y_masked
  %res = add nuw nsw <8 x i64> %mul, %z
  ret <8 x i64> %res
}

define <8 x i64> @test_512_no_combine_v2(<8 x i64> %x, <8 x i64> %y, <8 x i64> %z) {
  %mul = mul <8 x i64> %x, %y
  %res = add <8 x i64> %mul, %z
  ret <8 x i64> %res
}

define <4 x i64> @test_256_combine(<4 x i64> %x, <4 x i64> %y, <4 x i64> %z) {
  %x_masked = and <4 x i64> %x, splat(i64 67108863)
  %y_masked = and <4 x i64> %y, splat(i64 67108863)
  %mul = mul nuw nsw <4 x i64> %x_masked, %y_masked
  %res = add nuw nsw <4 x i64> %z, %mul
  ret <4 x i64> %res
}

define <4 x i64> @test_256_no_combine(<4 x i64> %x, <4 x i64> %y, <4 x i64> %z) {
  %mul = mul <4 x i64> %x, %y
  %res = add <4 x i64> %mul, %z
  ret <4 x i64> %res
}

define <2 x i64> @test_128_combine(<2 x i64> %x, <2 x i64> %y, <2 x i64> %z) {
  %x_masked = and <2 x i64> %x, splat (i64 67108863)
  %y_masked = and <2 x i64> %y, splat (i64 67108863)
  %mul = mul <2 x i64> %x_masked, %y_masked
  %res = add <2 x i64> %z, %mul
  ret <2 x i64> %res
}

; Sanity check we're not applying this here
define <1 x i64> @test_scalar_no_ifma(<1 x i64> %x, <1 x i64> %y, <1 x i64> %z) {
  %mul = mul <1 x i64> %x, %y
  %res = add <1 x i64> %mul, %z
  ret <1 x i64> %res
}

define <8 x i64> @test_mixed_width_too_wide(<8 x i64> %x, <8 x i64> %y, <8 x i64> %z) {
  ; 40-bit and 13-bit, too wide
  %x40 = and <8 x i64> %x, splat (i64 1099511627775)
  %y13 = and <8 x i64> %y, splat (i64 8191)
  %mul = mul <8 x i64> %x40, %y13
  %res = add <8 x i64> %z, %mul
  ret <8 x i64> %z
}

define <8 x i64> @test_zext32_inputs_not_safe(<8 x i32> %xi32, <8 x i32> %yi32, <8 x i64> %z) {
  %x = zext <8 x i32> %xi32 to <8 x i64>
  %y = zext <8 x i32> %yi32 to <8 x i64>
  %mul = mul <8 x i64> %x, %y
  %res = add <8 x i64> %z, %mul
  ret <8 x i64> %res
}

define <16 x i64> @test_1024_combine_split(<16 x i64> %x, <16 x i64> %y, <16 x i64> %z) {
  %x_masked = and <16 x i64> %x, splat (i64 67108863)
  %y_masked = and <16 x i64> %y, splat (i64 67108863)
  %mul = mul <16 x i64> %x_masked, %y_masked
  %res = add <16 x i64> %z, %mul
  ret <16 x i64> %res
}
