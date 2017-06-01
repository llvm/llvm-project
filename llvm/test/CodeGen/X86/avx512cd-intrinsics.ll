; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd | FileCheck %s

define <16 x i32> @test_x86_vbroadcastmw_512(i16 %a0) {
  ; CHECK: test_x86_vbroadcastmw_512
  ; CHECK: vpbroadcastmw2d %k0, %zmm0
  %res = call <16 x i32> @llvm.x86.avx512.broadcastmw.512(i16 %a0) ; 
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.broadcastmw.512(i16)

define <8 x i64> @test_x86_broadcastmb_512(i8 %a0) {
  ; CHECK: test_x86_broadcastmb_512
  ; CHECK: vpbroadcastmb2q %k0, %zmm0
  %res = call <8 x i64> @llvm.x86.avx512.broadcastmb.512(i8 %a0) ; 
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.broadcastmb.512(i8)

