; RUN: opt -passes=target-verifier -disable-output %s 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare <16 x float> @llvm.x86.avx512.max.ps.512(<16 x float>, <16 x float>, i32)
declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>)
declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>)
declare <4 x i32> @llvm.x86.avx512.conflict.d.128(<4 x i32>)
declare x86_amx @llvm.x86.tilezero.internal(i16, i16)

; CHECK: AVX-512 intrinsic used, but the subtarget does not support AVX-512.
define <16 x float> @avx512_without_feature(<16 x float> %a, <16 x float> %b) {
  %r = call <16 x float> @llvm.x86.avx512.max.ps.512(<16 x float> %a, <16 x float> %b, i32 4)
  ret <16 x float> %r
}

; CHECK: AVX intrinsic used, but the subtarget does not support AVX.
define <8 x float> @avx_without_feature(<8 x float> %a, <8 x float> %b) {
  %r = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a, <8 x float> %b)
  ret <8 x float> %r
}

; CHECK: AVX2 intrinsic used, but the subtarget does not support AVX2.
define <4 x i64> @avx2_without_feature(<32 x i8> %a, <32 x i8> %b) {
  %r = call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %a, <32 x i8> %b)
  ret <4 x i64> %r
}

; CHECK: x86_amx type used, but the subtarget does not support AMX-TILE.
define void @amx_without_feature() {
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 8)
  ret void
}

; CHECK: 128/256-bit AVX-512 intrinsic used, but the subtarget does not support AVX512VL.
define <4 x i32> @avx512vl_without_feature(<4 x i32> %a) #0 {
  %r = call <4 x i32> @llvm.x86.avx512.conflict.d.128(<4 x i32> %a)
  ret <4 x i32> %r
}

attributes #0 = { "target-features"="+avx512f" }
