; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=knl -mattr=-avx512cd -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX512 -check-prefix=AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skx -mattr=-avx512cd -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX512 -check-prefix=AVX512BW
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skx -mattr=+avx512cd -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX512CD

; Verify the cost of scalar leading zero count instructions.

declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare  i8 @llvm.ctlz.i8(i8, i1)

define i64 @var_ctlz_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i64':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i64 @llvm.ctlz.i64(i64 %a, i1 0)
  ret i64 %ctlz
}

define i64 @var_ctlz_i64u(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i64u':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i64 @llvm.ctlz.i64(i64 %a, i1 1)
  ret i64 %ctlz
}

define i32 @var_ctlz_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i32':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i32 @llvm.ctlz.i32(i32 %a, i1 0)
  ret i32 %ctlz
}

define i32 @var_ctlz_i32u(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i32u':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i32 @llvm.ctlz.i32(i32 %a, i1 1)
  ret i32 %ctlz
}

define i16 @var_ctlz_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i16':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i16 @llvm.ctlz.i16(i16 %a, i1 0)
  ret i16 %ctlz
}

define i16 @var_ctlz_i16u(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i16u':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i16 @llvm.ctlz.i16(i16 %a, i1 1)
  ret i16 %ctlz
}

define i8 @var_ctlz_i8(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i8':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i8 @llvm.ctlz.i8(i8 %a, i1 0)
  ret i8 %ctlz
}

define i8 @var_ctlz_i8u(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_i8u':
; CHECK: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call i8 @llvm.ctlz.i8(i8 %a, i1 1)
  ret i8 %ctlz
}

; Verify the cost of vector leading zero count instructions.

declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1)
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1)
declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1)

declare <4 x i64> @llvm.ctlz.v4i64(<4 x i64>, i1)
declare <8 x i32> @llvm.ctlz.v8i32(<8 x i32>, i1)
declare <16 x i16> @llvm.ctlz.v16i16(<16 x i16>, i1)
declare <32 x i8> @llvm.ctlz.v32i8(<32 x i8>, i1)

declare <8 x i64> @llvm.ctlz.v8i64(<8 x i64>, i1)
declare <16 x i32> @llvm.ctlz.v16i32(<16 x i32>, i1)
declare <32 x i16> @llvm.ctlz.v32i16(<32 x i16>, i1)
declare <64 x i8> @llvm.ctlz.v64i8(<64 x i8>, i1)

define <2 x i64> @var_ctlz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64':
; SSE2: Found an estimated cost of 25 for instruction:   %ctlz
; SSE42: Found an estimated cost of 23 for instruction:   %ctlz
; AVX: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %ctlz
}

define <2 x i64> @var_ctlz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64u':
; SSE2: Found an estimated cost of 25 for instruction:   %ctlz
; SSE42: Found an estimated cost of 23 for instruction:   %ctlz
; AVX: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64':
; SSE2: Found an estimated cost of 50 for instruction:   %ctlz
; SSE42: Found an estimated cost of 46 for instruction:   %ctlz
; AVX1: Found an estimated cost of 48 for instruction:   %ctlz
; AVX2: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64u':
; SSE2: Found an estimated cost of 50 for instruction:   %ctlz
; SSE42: Found an estimated cost of 46 for instruction:   %ctlz
; AVX1: Found an estimated cost of 48 for instruction:   %ctlz
; AVX2: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %ctlz
}

define <8 x i64> @var_ctlz_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i64':
; SSE2: Found an estimated cost of 100 for instruction:   %ctlz
; SSE42: Found an estimated cost of 92 for instruction:   %ctlz
; AVX1: Found an estimated cost of 96 for instruction:   %ctlz
; AVX2: Found an estimated cost of 46 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 29 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %a, i1 0)
  ret <8 x i64> %ctlz
}

define <8 x i64> @var_ctlz_v8i64u(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i64u':
; SSE2: Found an estimated cost of 100 for instruction:   %ctlz
; SSE42: Found an estimated cost of 92 for instruction:   %ctlz
; AVX1: Found an estimated cost of 96 for instruction:   %ctlz
; AVX2: Found an estimated cost of 46 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 29 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 23 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %a, i1 1)
  ret <8 x i64> %ctlz
}

define <4 x i32> @var_ctlz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32':
; SSE2: Found an estimated cost of 26 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %ctlz
}

define <4 x i32> @var_ctlz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32u':
; SSE2: Found an estimated cost of 26 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32':
; SSE2: Found an estimated cost of 52 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 38 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32u':
; SSE2: Found an estimated cost of 52 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 38 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %ctlz
}

define <16 x i32> @var_ctlz_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i32':
; SSE2: Found an estimated cost of 104 for instruction:   %ctlz
; SSE42: Found an estimated cost of 72 for instruction:   %ctlz
; AVX1: Found an estimated cost of 76 for instruction:   %ctlz
; AVX2: Found an estimated cost of 36 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 35 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 22 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a, i1 0)
  ret <16 x i32> %ctlz
}

define <16 x i32> @var_ctlz_v16i32u(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i32u':
; SSE2: Found an estimated cost of 104 for instruction:   %ctlz
; SSE42: Found an estimated cost of 72 for instruction:   %ctlz
; AVX1: Found an estimated cost of 76 for instruction:   %ctlz
; AVX2: Found an estimated cost of 36 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 35 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 22 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 1 for instruction:   %ctlz
  %ctlz = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a, i1 1)
  ret <16 x i32> %ctlz
}

define <8 x i16> @var_ctlz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16':
; SSE2: Found an estimated cost of 20 for instruction:   %ctlz
; SSE42: Found an estimated cost of 14 for instruction:   %ctlz
; AVX: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %ctlz
}

define <8 x i16> @var_ctlz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16u':
; SSE2: Found an estimated cost of 20 for instruction:   %ctlz
; SSE42: Found an estimated cost of 14 for instruction:   %ctlz
; AVX: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16':
; SSE2: Found an estimated cost of 40 for instruction:   %ctlz
; SSE42: Found an estimated cost of 28 for instruction:   %ctlz
; AVX1: Found an estimated cost of 30 for instruction:   %ctlz
; AVX2: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16u':
; SSE2: Found an estimated cost of 40 for instruction:   %ctlz
; SSE42: Found an estimated cost of 28 for instruction:   %ctlz
; AVX1: Found an estimated cost of 30 for instruction:   %ctlz
; AVX2: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512: Found an estimated cost of 14 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %ctlz
}

define <32 x i16> @var_ctlz_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i16':
; SSE2: Found an estimated cost of 80 for instruction:   %ctlz
; SSE42: Found an estimated cost of 56 for instruction:   %ctlz
; AVX1: Found an estimated cost of 60 for instruction:   %ctlz
; AVX2: Found an estimated cost of 28 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 28 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 8 for instruction:   %ctlz
  %ctlz = call <32 x i16> @llvm.ctlz.v32i16(<32 x i16> %a, i1 0)
  ret <32 x i16> %ctlz
}

define <32 x i16> @var_ctlz_v32i16u(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i16u':
; SSE2: Found an estimated cost of 80 for instruction:   %ctlz
; SSE42: Found an estimated cost of 56 for instruction:   %ctlz
; AVX1: Found an estimated cost of 60 for instruction:   %ctlz
; AVX2: Found an estimated cost of 28 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 28 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 8 for instruction:   %ctlz
  %ctlz = call <32 x i16> @llvm.ctlz.v32i16(<32 x i16> %a, i1 1)
  ret <32 x i16> %ctlz
}

define <16 x i8> @var_ctlz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8':
; SSE2: Found an estimated cost of 17 for instruction:   %ctlz
; SSE42: Found an estimated cost of 9 for instruction:   %ctlz
; AVX: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %ctlz
}

define <16 x i8> @var_ctlz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8u':
; SSE2: Found an estimated cost of 17 for instruction:   %ctlz
; SSE42: Found an estimated cost of 9 for instruction:   %ctlz
; AVX: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 4 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8':
; SSE2: Found an estimated cost of 34 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX1: Found an estimated cost of 20 for instruction:   %ctlz
; AVX2: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 10 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8u':
; SSE2: Found an estimated cost of 34 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX1: Found an estimated cost of 20 for instruction:   %ctlz
; AVX2: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512: Found an estimated cost of 9 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 10 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %ctlz
}

define <64 x i8> @var_ctlz_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v64i8':
; SSE2: Found an estimated cost of 68 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 40 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 17 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 20 for instruction:   %ctlz
  %ctlz = call <64 x i8> @llvm.ctlz.v64i8(<64 x i8> %a, i1 0)
  ret <64 x i8> %ctlz
}

define <64 x i8> @var_ctlz_v64i8u(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v64i8u':
; SSE2: Found an estimated cost of 68 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 40 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512F: Found an estimated cost of 18 for instruction:   %ctlz
; AVX512BW: Found an estimated cost of 17 for instruction:   %ctlz
; AVX512CD: Found an estimated cost of 20 for instruction:   %ctlz
  %ctlz = call <64 x i8> @llvm.ctlz.v64i8(<64 x i8> %a, i1 1)
  ret <64 x i8> %ctlz
}
