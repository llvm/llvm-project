; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+sse2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=SSE2
; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+sse4.2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=SSE42
; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=AVX
; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=AVX2
; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx512f -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=AVX512 -check-prefix=AVX512F
; RUN: opt < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx512vl,avx512bw,avx512dq -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X86 -check-prefix=AVX512 -check-prefix=AVX512BW
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse4.2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=AVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=AVX512 -check-prefix=AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512vl,+avx512bw,+avx512dq -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=X64 -check-prefix=AVX512 -check-prefix=AVX512BW
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+xop -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+xop,+avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX2

; Verify the cost of scalar bitreverse instructions.

declare i64 @llvm.bitreverse.i64(i64)
declare i32 @llvm.bitreverse.i32(i32)
declare i16 @llvm.bitreverse.i16(i16)
declare  i8 @llvm.bitreverse.i8(i8)

define i64 @var_bitreverse_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i64':
; X86: Found an estimated cost of 28 for instruction:   %bitreverse
; X64: Found an estimated cost of 14 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %bitreverse
}

define i32 @var_bitreverse_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i32':
; X86: Found an estimated cost of 14 for instruction:   %bitreverse
; X64: Found an estimated cost of 14 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %bitreverse
}

define i16 @var_bitreverse_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i16':
; X86: Found an estimated cost of 14 for instruction:   %bitreverse
; X64: Found an estimated cost of 14 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %bitreverse
}

define i8 @var_bitreverse_i8(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i8':
; X86: Found an estimated cost of 11 for instruction:   %bitreverse
; X64: Found an estimated cost of 11 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %bitreverse
}

; Verify the cost of vector bitreverse instructions.

declare <2 x i64> @llvm.bitreverse.v2i64(<2 x i64>)
declare <4 x i32> @llvm.bitreverse.v4i32(<4 x i32>)
declare <8 x i16> @llvm.bitreverse.v8i16(<8 x i16>)
declare <16 x i8> @llvm.bitreverse.v16i8(<16 x i8>)

declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)
declare <8 x i32> @llvm.bitreverse.v8i32(<8 x i32>)
declare <16 x i16> @llvm.bitreverse.v16i16(<16 x i16>)
declare <32 x i8> @llvm.bitreverse.v32i8(<32 x i8>)

declare <8 x i64> @llvm.bitreverse.v8i64(<8 x i64>)
declare <16 x i32> @llvm.bitreverse.v16i32(<16 x i32>)
declare <32 x i16> @llvm.bitreverse.v32i16(<32 x i16>)
declare <64 x i8> @llvm.bitreverse.v64i8(<64 x i8>)

define <2 x i64> @var_bitreverse_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v2i64':
; SSE2: Found an estimated cost of 29 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %a)
  ret <2 x i64> %bitreverse
}

define <4 x i64> @var_bitreverse_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v4i64':
; SSE2: Found an estimated cost of 58 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 12 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %a)
  ret <4 x i64> %bitreverse
}

define <8 x i64> @var_bitreverse_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v8i64':
; SSE2: Found an estimated cost of 116 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 20 for instruction:   %bitreverse
; AVX: Found an estimated cost of 24 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512F: Found an estimated cost of 36 for instruction:   %bitreverse
; AVX512BW: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 8 for instruction:   %bitreverse
  %bitreverse = call <8 x i64> @llvm.bitreverse.v8i64(<8 x i64> %a)
  ret <8 x i64> %bitreverse
}

define <4 x i32> @var_bitreverse_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v4i32':
; SSE2: Found an estimated cost of 27 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <4 x i32> @llvm.bitreverse.v4i32(<4 x i32> %a)
  ret <4 x i32> %bitreverse
}

define <8 x i32> @var_bitreverse_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v8i32':
; SSE2: Found an estimated cost of 54 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 12 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <8 x i32> @llvm.bitreverse.v8i32(<8 x i32> %a)
  ret <8 x i32> %bitreverse
}

define <16 x i32> @var_bitreverse_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v16i32':
; SSE2: Found an estimated cost of 108 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 20 for instruction:   %bitreverse
; AVX: Found an estimated cost of 24 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512F: Found an estimated cost of 24 for instruction:   %bitreverse
; AVX512BW: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 8 for instruction:   %bitreverse
  %bitreverse = call <16 x i32> @llvm.bitreverse.v16i32(<16 x i32> %a)
  ret <16 x i32> %bitreverse
}

define <8 x i16> @var_bitreverse_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v8i16':
; SSE2: Found an estimated cost of 27 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> %a)
  ret <8 x i16> %bitreverse
}

define <16 x i16> @var_bitreverse_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v16i16':
; SSE2: Found an estimated cost of 54 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 12 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <16 x i16> @llvm.bitreverse.v16i16(<16 x i16> %a)
  ret <16 x i16> %bitreverse
}

define <32 x i16> @var_bitreverse_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v32i16':
; SSE2: Found an estimated cost of 108 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 20 for instruction:   %bitreverse
; AVX: Found an estimated cost of 24 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512F: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512BW: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 8 for instruction:   %bitreverse
  %bitreverse = call <32 x i16> @llvm.bitreverse.v32i16(<32 x i16> %a)
  ret <32 x i16> %bitreverse
}

define <16 x i8> @var_bitreverse_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v16i8':
; SSE2: Found an estimated cost of 20 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> %a)
  ret <16 x i8> %bitreverse
}

define <32 x i8> @var_bitreverse_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v32i8':
; SSE2: Found an estimated cost of 40 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 12 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX512: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <32 x i8> @llvm.bitreverse.v32i8(<32 x i8> %a)
  ret <32 x i8> %bitreverse
}

define <64 x i8> @var_bitreverse_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v64i8':
; SSE2: Found an estimated cost of 80 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 20 for instruction:   %bitreverse
; AVX: Found an estimated cost of 24 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512F: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX512BW: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 8 for instruction:   %bitreverse
  %bitreverse = call <64 x i8> @llvm.bitreverse.v64i8(<64 x i8> %a)
  ret <64 x i8> %bitreverse
}
