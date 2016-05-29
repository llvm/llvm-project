; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+avx2 | FileCheck %s

define <16 x i16> @test_llvm_x86_avx2_pmovsxbw(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxbw
; CHECK: vpmovsxbw (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = sext <16 x i8> %1 to <16 x i16>
  ret <16 x i16> %2
}

define <8 x i32> @test_llvm_x86_avx2_pmovsxbd(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxbd
; CHECK: vpmovsxbd (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <4 x i64> @test_llvm_x86_avx2_pmovsxbq(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxbq
; CHECK: vpmovsxbq (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i32> @test_llvm_x86_avx2_pmovsxwd(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxwd
; CHECK: vpmovsxwd (%rdi), %ymm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = sext <8 x i16> %1 to <8 x i32>
  ret <8 x i32> %2
}

define <4 x i64> @test_llvm_x86_avx2_pmovsxwq(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxwq
; CHECK: vpmovsxwq (%rdi), %ymm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @test_llvm_x86_avx2_pmovsxdq(<4 x i32>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovsxdq
; CHECK: vpmovsxdq (%rdi), %ymm0
  %1 = load <4 x i32>, <4 x i32>* %a, align 1
  %2 = sext <4 x i32> %1 to <4 x i64>
  ret <4 x i64> %2
}

define <16 x i16> @test_llvm_x86_avx2_pmovzxbw(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxbw
; CHECK: vpmovzxbw (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = zext <16 x i8> %1 to <16 x i16>
  ret <16 x i16> %2
}

define <8 x i32> @test_llvm_x86_avx2_pmovzxbd(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxbd
; CHECK: vpmovzxbd (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = zext <8 x i8> %2 to <8 x i32>
  ret <8 x i32> %3
}

define <4 x i64> @test_llvm_x86_avx2_pmovzxbq(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxbq
; CHECK: vpmovzxbq (%rdi), %ymm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i8> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <8 x i32> @test_llvm_x86_avx2_pmovzxwd(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxwd
; CHECK: vpmovzxwd (%rdi), %ymm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = zext <8 x i16> %1 to <8 x i32>
  ret <8 x i32> %2
}

define <4 x i64> @test_llvm_x86_avx2_pmovzxwq(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxwq
; CHECK: vpmovzxwq (%rdi), %ymm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = zext <4 x i16> %2 to <4 x i64>
  ret <4 x i64> %3
}

define <4 x i64> @test_llvm_x86_avx2_pmovzxdq(<4 x i32>* %a) {
; CHECK-LABEL: test_llvm_x86_avx2_pmovzxdq
; CHECK: vpmovzxdq (%rdi), %ymm0
  %1 = load <4 x i32>, <4 x i32>* %a, align 1
  %2 = zext <4 x i32> %1 to <4 x i64>
  ret <4 x i64> %2
}
