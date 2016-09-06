; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX2

; Verify the cost of scalar bitreverse instructions.

declare i64 @llvm.bitreverse.i64(i64)
declare i32 @llvm.bitreverse.i32(i32)
declare i16 @llvm.bitreverse.i16(i16)
declare  i8 @llvm.bitreverse.i8(i8)

define i64 @var_bitreverse_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i64':
; SSE2: Found an estimated cost of 1 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 1 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %bitreverse
}

define i32 @var_bitreverse_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i32':
; SSE2: Found an estimated cost of 1 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 1 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %bitreverse
}

define i16 @var_bitreverse_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i16':
; SSE2: Found an estimated cost of 1 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 1 for instruction:   %bitreverse
; XOP: Found an estimated cost of 3 for instruction:   %bitreverse
  %bitreverse = call i16 @llvm.bitreverse.i16(i16 %a)
  ret i16 %bitreverse
}

define i8 @var_bitreverse_i8(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_i8':
; SSE2: Found an estimated cost of 1 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX: Found an estimated cost of 1 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 1 for instruction:   %bitreverse
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

define <2 x i64> @var_bitreverse_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v2i64':
; SSE2: Found an estimated cost of 6 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <2 x i64> @llvm.bitreverse.v2i64(<2 x i64> %a)
  ret <2 x i64> %bitreverse
}

define <4 x i64> @var_bitreverse_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v4i64':
; SSE2: Found an estimated cost of 12 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %a)
  ret <4 x i64> %bitreverse
}

define <4 x i32> @var_bitreverse_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v4i32':
; SSE2: Found an estimated cost of 12 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <4 x i32> @llvm.bitreverse.v4i32(<4 x i32> %a)
  ret <4 x i32> %bitreverse
}

define <8 x i32> @var_bitreverse_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v8i32':
; SSE2: Found an estimated cost of 24 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <8 x i32> @llvm.bitreverse.v8i32(<8 x i32> %a)
  ret <8 x i32> %bitreverse
}

define <8 x i16> @var_bitreverse_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v8i16':
; SSE2: Found an estimated cost of 24 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <8 x i16> @llvm.bitreverse.v8i16(<8 x i16> %a)
  ret <8 x i16> %bitreverse
}

define <16 x i16> @var_bitreverse_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v16i16':
; SSE2: Found an estimated cost of 48 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <16 x i16> @llvm.bitreverse.v16i16(<16 x i16> %a)
  ret <16 x i16> %bitreverse
}

define <16 x i8> @var_bitreverse_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v16i8':
; SSE2: Found an estimated cost of 48 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX: Found an estimated cost of 5 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 1 for instruction:   %bitreverse
  %bitreverse = call <16 x i8> @llvm.bitreverse.v16i8(<16 x i8> %a)
  ret <16 x i8> %bitreverse
}

define <32 x i8> @var_bitreverse_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bitreverse_v32i8':
; SSE2: Found an estimated cost of 96 for instruction:   %bitreverse
; SSE42: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX: Found an estimated cost of 10 for instruction:   %bitreverse
; AVX2: Found an estimated cost of 5 for instruction:   %bitreverse
; XOP: Found an estimated cost of 4 for instruction:   %bitreverse
  %bitreverse = call <32 x i8> @llvm.bitreverse.v32i8(<32 x i8> %a)
  ret <32 x i8> %bitreverse
}
