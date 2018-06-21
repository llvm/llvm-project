; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512dq | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512DQ

define <4 x i32> @test1(<4 x i32> %a) {
  %div = udiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test1':
; SSE: Found an estimated cost of 15 for instruction:   %div
; AVX: Found an estimated cost of 15 for instruction:   %div
; AVX512: Found an estimated cost of 15 for instruction:   %div
}

define <8 x i32> @test2(<8 x i32> %a) {
  %div = udiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test2':
; SSE: Found an estimated cost of 30 for instruction:   %div
; AVX1: Found an estimated cost of 30 for instruction:   %div
; AVX2: Found an estimated cost of 15 for instruction:   %div
; AVX512: Found an estimated cost of 15 for instruction:   %div
}

define <8 x i16> @test3(<8 x i16> %a) {
  %div = udiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; CHECK: 'Cost Model Analysis' for function 'test3':
; SSE: Found an estimated cost of 6 for instruction:   %div
; AVX: Found an estimated cost of 6 for instruction:   %div
; AVX512: Found an estimated cost of 6 for instruction:   %div
}

define <16 x i16> @test4(<16 x i16> %a) {
  %div = udiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; CHECK: 'Cost Model Analysis' for function 'test4':
; SSE: Found an estimated cost of 12 for instruction:   %div
; AVX1: Found an estimated cost of 12 for instruction:   %div
; AVX2: Found an estimated cost of 6 for instruction:   %div
; AVX512: Found an estimated cost of 6 for instruction:   %div
}

define <8 x i16> @test5(<8 x i16> %a) {
  %div = sdiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; CHECK: 'Cost Model Analysis' for function 'test5':
; SSE: Found an estimated cost of 6 for instruction:   %div
; AVX: Found an estimated cost of 6 for instruction:   %div
; AVX512: Found an estimated cost of 6 for instruction:   %div
}

define <16 x i16> @test6(<16 x i16> %a) {
  %div = sdiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; CHECK: 'Cost Model Analysis' for function 'test6':
; SSE: Found an estimated cost of 12 for instruction:   %div
; AVX1: Found an estimated cost of 12 for instruction:   %div
; AVX2: Found an estimated cost of 6 for instruction:   %div
; AVX512: Found an estimated cost of 6 for instruction:   %div
}

define <16 x i8> @test7(<16 x i8> %a) {
  %div = sdiv <16 x i8> %a, <i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %div

; CHECK: 'Cost Model Analysis' for function 'test7':
; SSE: Found an estimated cost of 320 for instruction:   %div
; AVX: Found an estimated cost of 320 for instruction:   %div
; AVX512: Found an estimated cost of 320 for instruction:   %div
}

define <4 x i32> @test8(<4 x i32> %a) {
  %div = sdiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test8':
; SSE: Found an estimated cost of 19 for instruction:   %div
; AVX: Found an estimated cost of 15 for instruction:   %div
; AVX512: Found an estimated cost of 15 for instruction:   %div
}

define <8 x i32> @test9(<8 x i32> %a) {
  %div = sdiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test9':
; SSE: Found an estimated cost of 38 for instruction:   %div
; AVX1: Found an estimated cost of 38 for instruction:   %div
; AVX2: Found an estimated cost of 15 for instruction:   %div
; AVX512: Found an estimated cost of 15 for instruction:   %div
}

define <8 x i32> @test10(<8 x i32> %a) {
  %div = sdiv <8 x i32> %a, <i32 8, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test10':
; SSE: Found an estimated cost of 160 for instruction:   %div
; AVX: Found an estimated cost of 160 for instruction:   %div
; AVX512: Found an estimated cost of 160 for instruction:   %div
}

define <16 x i32> @test11(<16 x i32> %a) {
  %div = sdiv <16 x i32> %a, <i32 8, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7, i32 8, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <16 x i32> %div

; CHECK: 'Cost Model Analysis' for function 'test11':
; SSE: Found an estimated cost of 320 for instruction:   %div
; AVX: Found an estimated cost of 320 for instruction:   %div
; AVX512: Found an estimated cost of 320 for instruction:   %div
}
