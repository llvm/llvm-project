; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX2

; Verify the cost of vector bswap instructions.

declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)

declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>)
declare <8 x i32> @llvm.bswap.v8i32(<8 x i32>)
declare <16 x i16> @llvm.bswap.v16i16(<16 x i16>)

define <2 x i64> @var_bswap_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v2i64':
; SSE2: Found an estimated cost of 7 for instruction:   %bswap
; SSE42: Found an estimated cost of 1 for instruction:   %bswap
; AVX: Found an estimated cost of 1 for instruction:   %bswap
; XOP: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %a)
  ret <2 x i64> %bswap
}

define <4 x i64> @var_bswap_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v4i64':
; SSE2: Found an estimated cost of 14 for instruction:   %bswap
; SSE42: Found an estimated cost of 2 for instruction:   %bswap
; AVX1: Found an estimated cost of 4 for instruction:   %bswap
; AVX2: Found an estimated cost of 1 for instruction:   %bswap
; XOPAVX1: Found an estimated cost of 4 for instruction:   %bswap
; XOPAVX2: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <4 x i64> @llvm.bswap.v4i64(<4 x i64> %a)
  ret <4 x i64> %bswap
}

define <4 x i32> @var_bswap_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v4i32':
; SSE2: Found an estimated cost of 7 for instruction:   %bswap
; SSE42: Found an estimated cost of 1 for instruction:   %bswap
; AVX: Found an estimated cost of 1 for instruction:   %bswap
; XOP: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %a)
  ret <4 x i32> %bswap
}

define <8 x i32> @var_bswap_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v8i32':
; SSE2: Found an estimated cost of 14 for instruction:   %bswap
; SSE42: Found an estimated cost of 2 for instruction:   %bswap
; AVX1: Found an estimated cost of 4 for instruction:   %bswap
; AVX2: Found an estimated cost of 1 for instruction:   %bswap
; XOPAVX1: Found an estimated cost of 4 for instruction:   %bswap
; XOPAVX2: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <8 x i32> @llvm.bswap.v8i32(<8 x i32> %a)
  ret <8 x i32> %bswap
}

define <8 x i16> @var_bswap_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v8i16':
; SSE2: Found an estimated cost of 7 for instruction:   %bswap
; SSE42: Found an estimated cost of 1 for instruction:   %bswap
; AVX: Found an estimated cost of 1 for instruction:   %bswap
; XOP: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %a)
  ret <8 x i16> %bswap
}

define <16 x i16> @var_bswap_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_bswap_v16i16':
; SSE2: Found an estimated cost of 14 for instruction:   %bswap
; SSE42: Found an estimated cost of 2 for instruction:   %bswap
; AVX1: Found an estimated cost of 4 for instruction:   %bswap
; AVX2: Found an estimated cost of 1 for instruction:   %bswap
; XOPAVX1: Found an estimated cost of 4 for instruction:   %bswap
; XOPAVX2: Found an estimated cost of 1 for instruction:   %bswap
  %bswap = call <16 x i16> @llvm.bswap.v16i16(<16 x i16> %a)
  ret <16 x i16> %bswap
}
