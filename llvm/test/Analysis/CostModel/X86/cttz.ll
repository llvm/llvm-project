; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=knl -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX512 -check-prefix=AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX512 -check-prefix=AVX512BW

; Verify the cost of scalar trailing zero count instructions.

declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare  i8 @llvm.cttz.i8(i8, i1)

define i64 @var_cttz_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i64':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i64 @llvm.cttz.i64(i64 %a, i1 0)
  ret i64 %cttz
}

define i64 @var_cttz_i64u(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i64u':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i64 @llvm.cttz.i64(i64 %a, i1 1)
  ret i64 %cttz
}

define i32 @var_cttz_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i32':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i32 @llvm.cttz.i32(i32 %a, i1 0)
  ret i32 %cttz
}

define i32 @var_cttz_i32u(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i32u':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i32 @llvm.cttz.i32(i32 %a, i1 1)
  ret i32 %cttz
}

define i16 @var_cttz_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i16':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i16 @llvm.cttz.i16(i16 %a, i1 0)
  ret i16 %cttz
}

define i16 @var_cttz_i16u(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i16u':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i16 @llvm.cttz.i16(i16 %a, i1 1)
  ret i16 %cttz
}

define i8 @var_cttz_i8(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i8':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i8 @llvm.cttz.i8(i8 %a, i1 0)
  ret i8 %cttz
}

define i8 @var_cttz_i8u(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_i8u':
; CHECK: Found an estimated cost of 1 for instruction:   %cttz
  %cttz = call i8 @llvm.cttz.i8(i8 %a, i1 1)
  ret i8 %cttz
}

; Verify the cost of vector trailing zero count instructions.

declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16>, i1)
declare <16 x i8> @llvm.cttz.v16i8(<16 x i8>, i1)

declare <4 x i64> @llvm.cttz.v4i64(<4 x i64>, i1)
declare <8 x i32> @llvm.cttz.v8i32(<8 x i32>, i1)
declare <16 x i16> @llvm.cttz.v16i16(<16 x i16>, i1)
declare <32 x i8> @llvm.cttz.v32i8(<32 x i8>, i1)

declare <8 x i64> @llvm.cttz.v8i64(<8 x i64>, i1)
declare <16 x i32> @llvm.cttz.v16i32(<16 x i32>, i1)
declare <32 x i16> @llvm.cttz.v32i16(<32 x i16>, i1)
declare <64 x i8> @llvm.cttz.v64i8(<64 x i8>, i1)

define <2 x i64> @var_cttz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64':
; SSE2: Found an estimated cost of 14 for instruction:   %cttz
; SSE42: Found an estimated cost of 10 for instruction:   %cttz
; AVX: Found an estimated cost of 10 for instruction:   %cttz
; AVX512: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %cttz
}

define <2 x i64> @var_cttz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64u':
; SSE2: Found an estimated cost of 14 for instruction:   %cttz
; SSE42: Found an estimated cost of 10 for instruction:   %cttz
; AVX: Found an estimated cost of 10 for instruction:   %cttz
; AVX512: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64':
; SSE2: Found an estimated cost of 28 for instruction:   %cttz
; SSE42: Found an estimated cost of 20 for instruction:   %cttz
; AVX1: Found an estimated cost of 22 for instruction:   %cttz
; AVX2: Found an estimated cost of 10 for instruction:   %cttz
; AVX512: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64u':
; SSE2: Found an estimated cost of 28 for instruction:   %cttz
; SSE42: Found an estimated cost of 20 for instruction:   %cttz
; AVX1: Found an estimated cost of 22 for instruction:   %cttz
; AVX2: Found an estimated cost of 10 for instruction:   %cttz
; AVX512: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %cttz
}

define <8 x i64> @var_cttz_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i64':
; SSE2: Found an estimated cost of 56 for instruction:   %cttz
; SSE42: Found an estimated cost of 40 for instruction:   %cttz
; AVX1: Found an estimated cost of 44 for instruction:   %cttz
; AVX2: Found an estimated cost of 20 for instruction:   %cttz
; AVX512F: Found an estimated cost of 20 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <8 x i64> @llvm.cttz.v8i64(<8 x i64> %a, i1 0)
  ret <8 x i64> %cttz
}

define <8 x i64> @var_cttz_v8i64u(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i64u':
; SSE2: Found an estimated cost of 56 for instruction:   %cttz
; SSE42: Found an estimated cost of 40 for instruction:   %cttz
; AVX1: Found an estimated cost of 44 for instruction:   %cttz
; AVX2: Found an estimated cost of 20 for instruction:   %cttz
; AVX512F: Found an estimated cost of 20 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <8 x i64> @llvm.cttz.v8i64(<8 x i64> %a, i1 1)
  ret <8 x i64> %cttz
}

define <4 x i32> @var_cttz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32':
; SSE2: Found an estimated cost of 18 for instruction:   %cttz
; SSE42: Found an estimated cost of 14 for instruction:   %cttz
; AVX: Found an estimated cost of 14 for instruction:   %cttz
; AVX512: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %cttz
}

define <4 x i32> @var_cttz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32u':
; SSE2: Found an estimated cost of 18 for instruction:   %cttz
; SSE42: Found an estimated cost of 14 for instruction:   %cttz
; AVX: Found an estimated cost of 14 for instruction:   %cttz
; AVX512: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32':
; SSE2: Found an estimated cost of 36 for instruction:   %cttz
; SSE42: Found an estimated cost of 28 for instruction:   %cttz
; AVX1: Found an estimated cost of 30 for instruction:   %cttz
; AVX2: Found an estimated cost of 14 for instruction:   %cttz
; AVX512: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32u':
; SSE2: Found an estimated cost of 36 for instruction:   %cttz
; SSE42: Found an estimated cost of 28 for instruction:   %cttz
; AVX1: Found an estimated cost of 30 for instruction:   %cttz
; AVX2: Found an estimated cost of 14 for instruction:   %cttz
; AVX512: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %cttz
}

define <16 x i32> @var_cttz_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i32':
; SSE2: Found an estimated cost of 72 for instruction:   %cttz
; SSE42: Found an estimated cost of 56 for instruction:   %cttz
; AVX1: Found an estimated cost of 60 for instruction:   %cttz
; AVX2: Found an estimated cost of 28 for instruction:   %cttz
; AVX512F: Found an estimated cost of 28 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <16 x i32> @llvm.cttz.v16i32(<16 x i32> %a, i1 0)
  ret <16 x i32> %cttz
}

define <16 x i32> @var_cttz_v16i32u(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i32u':
; SSE2: Found an estimated cost of 72 for instruction:   %cttz
; SSE42: Found an estimated cost of 56 for instruction:   %cttz
; AVX1: Found an estimated cost of 60 for instruction:   %cttz
; AVX2: Found an estimated cost of 28 for instruction:   %cttz
; AVX512F: Found an estimated cost of 28 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <16 x i32> @llvm.cttz.v16i32(<16 x i32> %a, i1 1)
  ret <16 x i32> %cttz
}

define <8 x i16> @var_cttz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16':
; SSE2: Found an estimated cost of 16 for instruction:   %cttz
; SSE42: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; AVX512: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %cttz
}

define <8 x i16> @var_cttz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16u':
; SSE2: Found an estimated cost of 16 for instruction:   %cttz
; SSE42: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; AVX512: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16':
; SSE2: Found an estimated cost of 32 for instruction:   %cttz
; SSE42: Found an estimated cost of 24 for instruction:   %cttz
; AVX1: Found an estimated cost of 26 for instruction:   %cttz
; AVX2: Found an estimated cost of 12 for instruction:   %cttz
; AVX512: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16u':
; SSE2: Found an estimated cost of 32 for instruction:   %cttz
; SSE42: Found an estimated cost of 24 for instruction:   %cttz
; AVX1: Found an estimated cost of 26 for instruction:   %cttz
; AVX2: Found an estimated cost of 12 for instruction:   %cttz
; AVX512: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %cttz
}

define <32 x i16> @var_cttz_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i16':
; SSE2: Found an estimated cost of 64 for instruction:   %cttz
; SSE42: Found an estimated cost of 48 for instruction:   %cttz
; AVX1: Found an estimated cost of 52 for instruction:   %cttz
; AVX2: Found an estimated cost of 24 for instruction:   %cttz
; AVX512F: Found an estimated cost of 24 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <32 x i16> @llvm.cttz.v32i16(<32 x i16> %a, i1 0)
  ret <32 x i16> %cttz
}

define <32 x i16> @var_cttz_v32i16u(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i16u':
; SSE2: Found an estimated cost of 64 for instruction:   %cttz
; SSE42: Found an estimated cost of 48 for instruction:   %cttz
; AVX1: Found an estimated cost of 52 for instruction:   %cttz
; AVX2: Found an estimated cost of 24 for instruction:   %cttz
; AVX512F: Found an estimated cost of 24 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <32 x i16> @llvm.cttz.v32i16(<32 x i16> %a, i1 1)
  ret <32 x i16> %cttz
}

define <16 x i8> @var_cttz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8':
; SSE2: Found an estimated cost of 13 for instruction:   %cttz
; SSE42: Found an estimated cost of 9 for instruction:   %cttz
; AVX: Found an estimated cost of 9 for instruction:   %cttz
; AVX512: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %cttz
}

define <16 x i8> @var_cttz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8u':
; SSE2: Found an estimated cost of 13 for instruction:   %cttz
; SSE42: Found an estimated cost of 9 for instruction:   %cttz
; AVX: Found an estimated cost of 9 for instruction:   %cttz
; AVX512: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8':
; SSE2: Found an estimated cost of 26 for instruction:   %cttz
; SSE42: Found an estimated cost of 18 for instruction:   %cttz
; AVX1: Found an estimated cost of 20 for instruction:   %cttz
; AVX2: Found an estimated cost of 9 for instruction:   %cttz
; AVX512: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8u':
; SSE2: Found an estimated cost of 26 for instruction:   %cttz
; SSE42: Found an estimated cost of 18 for instruction:   %cttz
; AVX1: Found an estimated cost of 20 for instruction:   %cttz
; AVX2: Found an estimated cost of 9 for instruction:   %cttz
; AVX512: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %cttz
}

define <64 x i8> @var_cttz_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v64i8':
; SSE2: Found an estimated cost of 52 for instruction:   %cttz
; SSE42: Found an estimated cost of 36 for instruction:   %cttz
; AVX1: Found an estimated cost of 40 for instruction:   %cttz
; AVX2: Found an estimated cost of 18 for instruction:   %cttz
; AVX512F: Found an estimated cost of 18 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <64 x i8> @llvm.cttz.v64i8(<64 x i8> %a, i1 0)
  ret <64 x i8> %cttz
}

define <64 x i8> @var_cttz_v64i8u(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v64i8u':
; SSE2: Found an estimated cost of 52 for instruction:   %cttz
; SSE42: Found an estimated cost of 36 for instruction:   %cttz
; AVX1: Found an estimated cost of 40 for instruction:   %cttz
; AVX2: Found an estimated cost of 18 for instruction:   %cttz
; AVX512F: Found an estimated cost of 18 for instruction:   %cttz
; AVX512BW: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <64 x i8> @llvm.cttz.v64i8(<64 x i8> %a, i1 1)
  ret <64 x i8> %cttz
}
