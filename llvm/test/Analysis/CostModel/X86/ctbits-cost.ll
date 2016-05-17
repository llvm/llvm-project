; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX2

; Verify the cost of vector population count instructions.

declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>)
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>)

declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>)
declare <32 x i8> @llvm.ctpop.v32i8(<32 x i8>)

define <2 x i64> @var_ctpop_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v2i64':
; SSE: Found an estimated cost of 2 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %ctpop
}

define <4 x i64> @var_ctpop_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v4i64':
; SSE: Found an estimated cost of 4 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %a)
  ret <4 x i64> %ctpop
}

define <4 x i32> @var_ctpop_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v4i32':
; SSE: Found an estimated cost of 2 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
  ret <4 x i32> %ctpop
}

define <8 x i32> @var_ctpop_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v8i32':
; SSE: Found an estimated cost of 4 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %a)
  ret <8 x i32> %ctpop
}

define <8 x i16> @var_ctpop_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v8i16':
; SSE: Found an estimated cost of 2 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
  ret <8 x i16> %ctpop
}

define <16 x i16> @var_ctpop_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v16i16':
; SSE: Found an estimated cost of 4 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %a)
  ret <16 x i16> %ctpop
}

define <16 x i8> @var_ctpop_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v16i8':
; SSE: Found an estimated cost of 2 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
  ret <16 x i8> %ctpop
}

define <32 x i8> @var_ctpop_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v32i8':
; SSE: Found an estimated cost of 4 for instruction:   %ctpop
; AVX: Found an estimated cost of 2 for instruction:   %ctpop
; XOP: Found an estimated cost of 2 for instruction:   %ctpop
  %ctpop = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %a)
  ret <32 x i8> %ctpop
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

define <2 x i64> @var_ctlz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64':
; SSE: Found an estimated cost of 6 for instruction:   %ctlz
; AVX: Found an estimated cost of 6 for instruction:   %ctlz
; XOP: Found an estimated cost of 6 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %ctlz
}

define <2 x i64> @var_ctlz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64u':
; SSE: Found an estimated cost of 6 for instruction:   %ctlz
; AVX: Found an estimated cost of 6 for instruction:   %ctlz
; XOP: Found an estimated cost of 6 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64':
; SSE: Found an estimated cost of 12 for instruction:   %ctlz
; AVX: Found an estimated cost of 12 for instruction:   %ctlz
; XOP: Found an estimated cost of 12 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64u':
; SSE: Found an estimated cost of 12 for instruction:   %ctlz
; AVX: Found an estimated cost of 12 for instruction:   %ctlz
; XOP: Found an estimated cost of 12 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %ctlz
}

define <4 x i32> @var_ctlz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32':
; SSE: Found an estimated cost of 12 for instruction:   %ctlz
; AVX: Found an estimated cost of 12 for instruction:   %ctlz
; XOP: Found an estimated cost of 12 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %ctlz
}

define <4 x i32> @var_ctlz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32u':
; SSE: Found an estimated cost of 12 for instruction:   %ctlz
; AVX: Found an estimated cost of 12 for instruction:   %ctlz
; XOP: Found an estimated cost of 12 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32':
; SSE: Found an estimated cost of 24 for instruction:   %ctlz
; AVX: Found an estimated cost of 24 for instruction:   %ctlz
; XOP: Found an estimated cost of 24 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32u':
; SSE: Found an estimated cost of 24 for instruction:   %ctlz
; AVX: Found an estimated cost of 24 for instruction:   %ctlz
; XOP: Found an estimated cost of 24 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %ctlz
}

define <8 x i16> @var_ctlz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16':
; SSE: Found an estimated cost of 24 for instruction:   %ctlz
; AVX: Found an estimated cost of 24 for instruction:   %ctlz
; XOP: Found an estimated cost of 24 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %ctlz
}

define <8 x i16> @var_ctlz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16u':
; SSE: Found an estimated cost of 24 for instruction:   %ctlz
; AVX: Found an estimated cost of 24 for instruction:   %ctlz
; XOP: Found an estimated cost of 24 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16':
; SSE: Found an estimated cost of 48 for instruction:   %ctlz
; AVX: Found an estimated cost of 48 for instruction:   %ctlz
; XOP: Found an estimated cost of 48 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16u':
; SSE: Found an estimated cost of 48 for instruction:   %ctlz
; AVX: Found an estimated cost of 48 for instruction:   %ctlz
; XOP: Found an estimated cost of 48 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %ctlz
}

define <16 x i8> @var_ctlz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8':
; SSE: Found an estimated cost of 48 for instruction:   %ctlz
; AVX: Found an estimated cost of 48 for instruction:   %ctlz
; XOP: Found an estimated cost of 48 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %ctlz
}

define <16 x i8> @var_ctlz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8u':
; SSE: Found an estimated cost of 48 for instruction:   %ctlz
; AVX: Found an estimated cost of 48 for instruction:   %ctlz
; XOP: Found an estimated cost of 48 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8':
; SSE: Found an estimated cost of 96 for instruction:   %ctlz
; AVX: Found an estimated cost of 96 for instruction:   %ctlz
; XOP: Found an estimated cost of 96 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8u':
; SSE: Found an estimated cost of 96 for instruction:   %ctlz
; AVX: Found an estimated cost of 96 for instruction:   %ctlz
; XOP: Found an estimated cost of 96 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %ctlz
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

define <2 x i64> @var_cttz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64':
; SSE: Found an estimated cost of 6 for instruction:   %cttz
; AVX: Found an estimated cost of 6 for instruction:   %cttz
; XOP: Found an estimated cost of 6 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %cttz
}

define <2 x i64> @var_cttz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64u':
; SSE: Found an estimated cost of 6 for instruction:   %cttz
; AVX: Found an estimated cost of 6 for instruction:   %cttz
; XOP: Found an estimated cost of 6 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64':
; SSE: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; XOP: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64u':
; SSE: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; XOP: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %cttz
}

define <4 x i32> @var_cttz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32':
; SSE: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; XOP: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %cttz
}

define <4 x i32> @var_cttz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32u':
; SSE: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
; XOP: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32':
; SSE: Found an estimated cost of 24 for instruction:   %cttz
; AVX: Found an estimated cost of 24 for instruction:   %cttz
; XOP: Found an estimated cost of 24 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32u':
; SSE: Found an estimated cost of 24 for instruction:   %cttz
; AVX: Found an estimated cost of 24 for instruction:   %cttz
; XOP: Found an estimated cost of 24 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %cttz
}

define <8 x i16> @var_cttz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16':
; SSE: Found an estimated cost of 24 for instruction:   %cttz
; AVX: Found an estimated cost of 24 for instruction:   %cttz
; XOP: Found an estimated cost of 24 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %cttz
}

define <8 x i16> @var_cttz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16u':
; SSE: Found an estimated cost of 24 for instruction:   %cttz
; AVX: Found an estimated cost of 24 for instruction:   %cttz
; XOP: Found an estimated cost of 24 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16':
; SSE: Found an estimated cost of 48 for instruction:   %cttz
; AVX: Found an estimated cost of 48 for instruction:   %cttz
; XOP: Found an estimated cost of 48 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16u':
; SSE: Found an estimated cost of 48 for instruction:   %cttz
; AVX: Found an estimated cost of 48 for instruction:   %cttz
; XOP: Found an estimated cost of 48 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %cttz
}

define <16 x i8> @var_cttz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8':
; SSE: Found an estimated cost of 48 for instruction:   %cttz
; AVX: Found an estimated cost of 48 for instruction:   %cttz
; XOP: Found an estimated cost of 48 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %cttz
}

define <16 x i8> @var_cttz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8u':
; SSE: Found an estimated cost of 48 for instruction:   %cttz
; AVX: Found an estimated cost of 48 for instruction:   %cttz
; XOP: Found an estimated cost of 48 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8':
; SSE: Found an estimated cost of 96 for instruction:   %cttz
; AVX: Found an estimated cost of 96 for instruction:   %cttz
; XOP: Found an estimated cost of 96 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8u':
; SSE: Found an estimated cost of 96 for instruction:   %cttz
; AVX: Found an estimated cost of 96 for instruction:   %cttz
; XOP: Found an estimated cost of 96 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %cttz
}
