; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=pentium4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2 -check-prefix=NOPOPCNT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42 -check-prefix=POPCNT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1 -check-prefix=POPCNT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2 -check-prefix=POPCNT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1 -check-prefix=POPCNT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2 -check-prefix=POPCNT

; Verify the cost of scalar population count instructions.

declare i64 @llvm.ctpop.i64(i64)
declare i32 @llvm.ctpop.i32(i32)
declare i16 @llvm.ctpop.i16(i16)
declare  i8 @llvm.ctpop.i8(i8)

define i64 @var_ctpop_i64(i64 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_i64':
; NOPOPCNT: Found an estimated cost of 4 for instruction:   %ctpop
; POPCNT: Found an estimated cost of 1 for instruction:   %ctpop
  %ctpop = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %ctpop
}

define i32 @var_ctpop_i32(i32 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_i32':
; NOPOPCNT: Found an estimated cost of 4 for instruction:   %ctpop
; POPCNT: Found an estimated cost of 1 for instruction:   %ctpop
  %ctpop = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %ctpop
}

define i16 @var_ctpop_i16(i16 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_i16':
; NOPOPCNT: Found an estimated cost of 4 for instruction:   %ctpop
; POPCNT: Found an estimated cost of 1 for instruction:   %ctpop
  %ctpop = call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %ctpop
}

define i8 @var_ctpop_i8(i8 %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_i8':
; NOPOPCNT: Found an estimated cost of 4 for instruction:   %ctpop
; POPCNT: Found an estimated cost of 1 for instruction:   %ctpop
  %ctpop = call i8 @llvm.ctpop.i8(i8 %a)
  ret i8 %ctpop
}

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
; SSE2: Found an estimated cost of 12 for instruction:   %ctpop
; SSE42: Found an estimated cost of 7 for instruction:   %ctpop
; AVX: Found an estimated cost of 7 for instruction:   %ctpop
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %ctpop
}

define <4 x i64> @var_ctpop_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v4i64':
; SSE2: Found an estimated cost of 24 for instruction:   %ctpop
; SSE42: Found an estimated cost of 14 for instruction:   %ctpop
; AVX1: Found an estimated cost of 16 for instruction:   %ctpop
; AVX2: Found an estimated cost of 7 for instruction:   %ctpop
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %a)
  ret <4 x i64> %ctpop
}

define <4 x i32> @var_ctpop_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v4i32':
; SSE2: Found an estimated cost of 15 for instruction:   %ctpop
; SSE42: Found an estimated cost of 11 for instruction:   %ctpop
; AVX: Found an estimated cost of 11 for instruction:   %ctpop
  %ctpop = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
  ret <4 x i32> %ctpop
}

define <8 x i32> @var_ctpop_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v8i32':
; SSE2: Found an estimated cost of 30 for instruction:   %ctpop
; SSE42: Found an estimated cost of 22 for instruction:   %ctpop
; AVX1: Found an estimated cost of 24 for instruction:   %ctpop
; AVX2: Found an estimated cost of 11 for instruction:   %ctpop
  %ctpop = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %a)
  ret <8 x i32> %ctpop
}

define <8 x i16> @var_ctpop_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v8i16':
; SSE2: Found an estimated cost of 13 for instruction:   %ctpop
; SSE42: Found an estimated cost of 9 for instruction:   %ctpop
; AVX: Found an estimated cost of 9 for instruction:   %ctpop
  %ctpop = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
  ret <8 x i16> %ctpop
}

define <16 x i16> @var_ctpop_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v16i16':
; SSE2: Found an estimated cost of 26 for instruction:   %ctpop
; SSE42: Found an estimated cost of 18 for instruction:   %ctpop
; AVX1: Found an estimated cost of 20 for instruction:   %ctpop
; AVX2: Found an estimated cost of 9 for instruction:   %ctpop
  %ctpop = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %a)
  ret <16 x i16> %ctpop
}

define <16 x i8> @var_ctpop_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v16i8':
; SSE2: Found an estimated cost of 10 for instruction:   %ctpop
; SSE42: Found an estimated cost of 6 for instruction:   %ctpop
; AVX: Found an estimated cost of 6 for instruction:   %ctpop
  %ctpop = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
  ret <16 x i8> %ctpop
}

define <32 x i8> @var_ctpop_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctpop_v32i8':
; SSE2: Found an estimated cost of 20 for instruction:   %ctpop
; SSE42: Found an estimated cost of 12 for instruction:   %ctpop
; AVX1: Found an estimated cost of 14 for instruction:   %ctpop
; AVX2: Found an estimated cost of 6 for instruction:   %ctpop
  %ctpop = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %a)
  ret <32 x i8> %ctpop
}

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

define <2 x i64> @var_ctlz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64':
; SSE2: Found an estimated cost of 25 for instruction:   %ctlz
; SSE42: Found an estimated cost of 23 for instruction:   %ctlz
; AVX: Found an estimated cost of 23 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %ctlz
}

define <2 x i64> @var_ctlz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v2i64u':
; SSE2: Found an estimated cost of 25 for instruction:   %ctlz
; SSE42: Found an estimated cost of 23 for instruction:   %ctlz
; AVX: Found an estimated cost of 23 for instruction:   %ctlz
  %ctlz = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64':
; SSE2: Found an estimated cost of 50 for instruction:   %ctlz
; SSE42: Found an estimated cost of 46 for instruction:   %ctlz
; AVX1: Found an estimated cost of 48 for instruction:   %ctlz
; AVX2: Found an estimated cost of 23 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %ctlz
}

define <4 x i64> @var_ctlz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i64u':
; SSE2: Found an estimated cost of 50 for instruction:   %ctlz
; SSE42: Found an estimated cost of 46 for instruction:   %ctlz
; AVX1: Found an estimated cost of 48 for instruction:   %ctlz
; AVX2: Found an estimated cost of 23 for instruction:   %ctlz
  %ctlz = call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %ctlz
}

define <4 x i32> @var_ctlz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32':
; SSE2: Found an estimated cost of 26 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX: Found an estimated cost of 18 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %ctlz
}

define <4 x i32> @var_ctlz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v4i32u':
; SSE2: Found an estimated cost of 26 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX: Found an estimated cost of 18 for instruction:   %ctlz
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32':
; SSE2: Found an estimated cost of 52 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 38 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %ctlz
}

define <8 x i32> @var_ctlz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i32u':
; SSE2: Found an estimated cost of 52 for instruction:   %ctlz
; SSE42: Found an estimated cost of 36 for instruction:   %ctlz
; AVX1: Found an estimated cost of 38 for instruction:   %ctlz
; AVX2: Found an estimated cost of 18 for instruction:   %ctlz
  %ctlz = call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %ctlz
}

define <8 x i16> @var_ctlz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16':
; SSE2: Found an estimated cost of 20 for instruction:   %ctlz
; SSE42: Found an estimated cost of 14 for instruction:   %ctlz
; AVX: Found an estimated cost of 14 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %ctlz
}

define <8 x i16> @var_ctlz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v8i16u':
; SSE2: Found an estimated cost of 20 for instruction:   %ctlz
; SSE42: Found an estimated cost of 14 for instruction:   %ctlz
; AVX: Found an estimated cost of 14 for instruction:   %ctlz
  %ctlz = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16':
; SSE2: Found an estimated cost of 40 for instruction:   %ctlz
; SSE42: Found an estimated cost of 28 for instruction:   %ctlz
; AVX1: Found an estimated cost of 30 for instruction:   %ctlz
; AVX2: Found an estimated cost of 14 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %ctlz
}

define <16 x i16> @var_ctlz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i16u':
; SSE2: Found an estimated cost of 40 for instruction:   %ctlz
; SSE42: Found an estimated cost of 28 for instruction:   %ctlz
; AVX1: Found an estimated cost of 30 for instruction:   %ctlz
; AVX2: Found an estimated cost of 14 for instruction:   %ctlz
  %ctlz = call <16 x i16> @llvm.ctlz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %ctlz
}

define <16 x i8> @var_ctlz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8':
; SSE2: Found an estimated cost of 17 for instruction:   %ctlz
; SSE42: Found an estimated cost of 9 for instruction:   %ctlz
; AVX: Found an estimated cost of 9 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %ctlz
}

define <16 x i8> @var_ctlz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v16i8u':
; SSE2: Found an estimated cost of 17 for instruction:   %ctlz
; SSE42: Found an estimated cost of 9 for instruction:   %ctlz
; AVX: Found an estimated cost of 9 for instruction:   %ctlz
  %ctlz = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8':
; SSE2: Found an estimated cost of 34 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX1: Found an estimated cost of 20 for instruction:   %ctlz
; AVX2: Found an estimated cost of 9 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %ctlz
}

define <32 x i8> @var_ctlz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_ctlz_v32i8u':
; SSE2: Found an estimated cost of 34 for instruction:   %ctlz
; SSE42: Found an estimated cost of 18 for instruction:   %ctlz
; AVX1: Found an estimated cost of 20 for instruction:   %ctlz
; AVX2: Found an estimated cost of 9 for instruction:   %ctlz
  %ctlz = call <32 x i8> @llvm.ctlz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %ctlz
}

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

define <2 x i64> @var_cttz_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64':
; SSE2: Found an estimated cost of 14 for instruction:   %cttz
; SSE42: Found an estimated cost of 10 for instruction:   %cttz
; AVX: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 0)
  ret <2 x i64> %cttz
}

define <2 x i64> @var_cttz_v2i64u(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v2i64u':
; SSE2: Found an estimated cost of 14 for instruction:   %cttz
; SSE42: Found an estimated cost of 10 for instruction:   %cttz
; AVX: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 1)
  ret <2 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64':
; SSE2: Found an estimated cost of 28 for instruction:   %cttz
; SSE42: Found an estimated cost of 20 for instruction:   %cttz
; AVX1: Found an estimated cost of 22 for instruction:   %cttz
; AVX2: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 0)
  ret <4 x i64> %cttz
}

define <4 x i64> @var_cttz_v4i64u(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i64u':
; SSE2: Found an estimated cost of 28 for instruction:   %cttz
; SSE42: Found an estimated cost of 20 for instruction:   %cttz
; AVX1: Found an estimated cost of 22 for instruction:   %cttz
; AVX2: Found an estimated cost of 10 for instruction:   %cttz
  %cttz = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %a, i1 1)
  ret <4 x i64> %cttz
}

define <4 x i32> @var_cttz_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32':
; SSE2: Found an estimated cost of 18 for instruction:   %cttz
; SSE42: Found an estimated cost of 14 for instruction:   %cttz
; AVX: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 0)
  ret <4 x i32> %cttz
}

define <4 x i32> @var_cttz_v4i32u(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v4i32u':
; SSE2: Found an estimated cost of 18 for instruction:   %cttz
; SSE42: Found an estimated cost of 14 for instruction:   %cttz
; AVX: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 1)
  ret <4 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32':
; SSE2: Found an estimated cost of 36 for instruction:   %cttz
; SSE42: Found an estimated cost of 28 for instruction:   %cttz
; AVX1: Found an estimated cost of 30 for instruction:   %cttz
; AVX2: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 0)
  ret <8 x i32> %cttz
}

define <8 x i32> @var_cttz_v8i32u(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i32u':
; SSE2: Found an estimated cost of 36 for instruction:   %cttz
; SSE42: Found an estimated cost of 28 for instruction:   %cttz
; AVX1: Found an estimated cost of 30 for instruction:   %cttz
; AVX2: Found an estimated cost of 14 for instruction:   %cttz
  %cttz = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %a, i1 1)
  ret <8 x i32> %cttz
}

define <8 x i16> @var_cttz_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16':
; SSE2: Found an estimated cost of 16 for instruction:   %cttz
; SSE42: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 0)
  ret <8 x i16> %cttz
}

define <8 x i16> @var_cttz_v8i16u(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v8i16u':
; SSE2: Found an estimated cost of 16 for instruction:   %cttz
; SSE42: Found an estimated cost of 12 for instruction:   %cttz
; AVX: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 1)
  ret <8 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16':
; SSE2: Found an estimated cost of 32 for instruction:   %cttz
; SSE42: Found an estimated cost of 24 for instruction:   %cttz
; AVX1: Found an estimated cost of 26 for instruction:   %cttz
; AVX2: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 0)
  ret <16 x i16> %cttz
}

define <16 x i16> @var_cttz_v16i16u(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i16u':
; SSE2: Found an estimated cost of 32 for instruction:   %cttz
; SSE42: Found an estimated cost of 24 for instruction:   %cttz
; AVX1: Found an estimated cost of 26 for instruction:   %cttz
; AVX2: Found an estimated cost of 12 for instruction:   %cttz
  %cttz = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %a, i1 1)
  ret <16 x i16> %cttz
}

define <16 x i8> @var_cttz_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8':
; SSE2: Found an estimated cost of 13 for instruction:   %cttz
; SSE42: Found an estimated cost of 9 for instruction:   %cttz
; AVX: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 0)
  ret <16 x i8> %cttz
}

define <16 x i8> @var_cttz_v16i8u(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v16i8u':
; SSE2: Found an estimated cost of 13 for instruction:   %cttz
; SSE42: Found an estimated cost of 9 for instruction:   %cttz
; AVX: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 1)
  ret <16 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8':
; SSE2: Found an estimated cost of 26 for instruction:   %cttz
; SSE42: Found an estimated cost of 18 for instruction:   %cttz
; AVX1: Found an estimated cost of 20 for instruction:   %cttz
; AVX2: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 0)
  ret <32 x i8> %cttz
}

define <32 x i8> @var_cttz_v32i8u(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'var_cttz_v32i8u':
; SSE2: Found an estimated cost of 26 for instruction:   %cttz
; SSE42: Found an estimated cost of 18 for instruction:   %cttz
; AVX1: Found an estimated cost of 20 for instruction:   %cttz
; AVX2: Found an estimated cost of 9 for instruction:   %cttz
  %cttz = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %a, i1 1)
  ret <32 x i8> %cttz
}
