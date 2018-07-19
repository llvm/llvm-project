; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX1
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'trunc_vXi32'
define i32 @trunc_vXi32() {
  ; SSE: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX1: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX2: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX512: cost of 0 {{.*}} %V2i64 = trunc
  %V2i64 = trunc <2 x i64> undef to <2 x i32>

  ; SSE: cost of 1 {{.*}} %V4i64 = trunc
  ; AVX1: cost of 4 {{.*}} %V4i64 = trunc
  ; AVX2: cost of 2 {{.*}} %V4i64 = trunc
  ; AVX512: cost of 2 {{.*}} %V4i64 = trunc
  %V4i64 = trunc <4 x i64> undef to <4 x i32>

  ; SSE: cost of 3 {{.*}} %V8i64 = trunc
  ; AVX1: cost of 9 {{.*}} %V8i64 = trunc
  ; AVX2: cost of 4 {{.*}} %V8i64 = trunc
  ; AVX512: cost of 1 {{.*}} %V8i64 = trunc
  %V8i64 = trunc <8 x i64> undef to <8 x i32>

  ret i32 undef
}

; CHECK-LABEL: 'trunc_vXi16'
define i32 @trunc_vXi16() {
  ; SSE: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX512: cost of 0 {{.*}} %V2i64 = trunc
  %V2i64 = trunc <2 x i64> undef to <2 x i16>

  ; SSE: cost of 1 {{.*}} %V4i64 = trunc
  ; AVX1: cost of 4 {{.*}} %V4i64 = trunc
  ; AVX2: cost of 2 {{.*}} %V4i64 = trunc
  ; AVX512: cost of 2 {{.*}} %V4i64 = trunc
  %V4i64 = trunc <4 x i64> undef to <4 x i16>

  ; SSE: cost of 3 {{.*}} %V8i64 = trunc
  ; AVX: cost of 0 {{.*}} %V8i64 = trunc
  ; AVX512: cost of 1 {{.*}} %V8i64 = trunc
  %V8i64 = trunc <8 x i64> undef to <8 x i16>

  ; SSE2: cost of 3 {{.*}} %V4i32 = trunc
  ; SSSE3: cost of 3 {{.*}} %V4i32 = trunc
  ; SSE42: cost of 1 {{.*}} %V4i32 = trunc
  ; AVX1: cost of 1 {{.*}} %V4i32 = trunc
  ; AVX2: cost of 1 {{.*}} %V4i32 = trunc
  ; AVX512: cost of 1 {{.*}} %V4i32 = trunc
  %V4i32 = trunc <4 x i32> undef to <4 x i16>

  ; SSE2: cost of 5 {{.*}} %V8i32 = trunc
  ; SSSE3: cost of 5 {{.*}} %V8i32 = trunc
  ; SSE42: cost of 3 {{.*}} %V8i32 = trunc
  ; AVX1: cost of 5 {{.*}} %V8i32 = trunc
  ; AVX2: cost of 2 {{.*}} %V8i32 = trunc
  ; AVX512: cost of 2 {{.*}} %V8i32 = trunc
  %V8i32 = trunc <8 x i32> undef to <8 x i16>

  ; SSE2: cost of 10 {{.*}} %V16i32 = trunc
  ; SSSE3: cost of 10 {{.*}} %V16i32 = trunc
  ; SSE42: cost of 6 {{.*}} %V16i32 = trunc
  ; AVX1: cost of 6 {{.*}} %V16i32 = trunc
  ; AVX2: cost of 6 {{.*}} %V16i32 = trunc
  ; AVX512: cost of 1 {{.*}} %V16i32 = trunc
  %V16i32 = trunc <16 x i32> undef to <16 x i16>

  ret i32 undef
}

; CHECK-LABEL: 'trunc_vXi8'
define i32 @trunc_vXi8() {
  ; SSE: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX: cost of 0 {{.*}} %V2i64 = trunc
  ; AVX512: cost of 0 {{.*}} %V2i64 = trunc
  %V2i64 = trunc <2 x i64> undef to <2 x i8>

  ; SSE: cost of 1 {{.*}} %V4i64 = trunc
  ; AVX1: cost of 4 {{.*}} %V4i64 = trunc
  ; AVX2: cost of 2 {{.*}} %V4i64 = trunc
  ; AVX512: cost of 2 {{.*}} %V4i64 = trunc
  %V4i64 = trunc <4 x i64> undef to <4 x i8>

  ; SSE: cost of 3 {{.*}} %V8i64 = trunc
  ; AVX: cost of 0 {{.*}} %V8i64 = trunc
  ; AVX512: cost of 0 {{.*}} %V8i64 = trunc
  %V8i64 = trunc <8 x i64> undef to <8 x i8>

  ; SSE: cost of 0 {{.*}} %V2i32 = trunc
  ; AVX: cost of 0 {{.*}} %V2i32 = trunc
  ; AVX512: cost of 0 {{.*}} %V2i32 = trunc
  %V2i32 = trunc <2 x i32> undef to <2 x i8>

  ; SSE2: cost of 3 {{.*}} %V4i32 = trunc
  ; SSSE3: cost of 3 {{.*}} %V4i32 = trunc
  ; SSE42: cost of 1 {{.*}} %V4i32 = trunc
  ; AVX: cost of 1 {{.*}} %V4i32 = trunc
  ; AVX512: cost of 1 {{.*}} %V4i32 = trunc
  %V4i32 = trunc <4 x i32> undef to <4 x i8>

  ; SSE2: cost of 4 {{.*}} %V8i32 = trunc
  ; SSSE3: cost of 4 {{.*}} %V8i32 = trunc
  ; SSE42: cost of 3 {{.*}} %V8i32 = trunc
  ; AVX1: cost of 4 {{.*}} %V8i32 = trunc
  ; AVX2: cost of 2 {{.*}} %V8i32 = trunc
  ; AVX512: cost of 2 {{.*}} %V8i32 = trunc
  %V8i32 = trunc <8 x i32> undef to <8 x i8>

  ; SSE: cost of 7 {{.*}} %V16i32 = trunc
  ; AVX: cost of 7 {{.*}} %V16i32 = trunc
  ; AVX512: cost of 1 {{.*}} %V16i32 = trunc
  %V16i32 = trunc <16 x i32> undef to <16 x i8>

  ; SSE: cost of 0 {{.*}} %V2i16 = trunc
  ; AVX: cost of 0 {{.*}} %V2i16 = trunc
  ; AVX512: cost of 0 {{.*}} %V2i16 = trunc
  %V2i16 = trunc <2 x i16> undef to <2 x i8>

  ; SSE2: cost of 4 {{.*}} %V4i16 = trunc
  ; SSSE3: cost of 4 {{.*}} %V4i16 = trunc
  ; SSE42: cost of 2 {{.*}} %V4i16 = trunc
  ; AVX: cost of 2 {{.*}} %V4i16 = trunc
  ; AVX512: cost of 2 {{.*}} %V4i16 = trunc
  %V4i16 = trunc <4 x i16> undef to <4 x i8>

  ; SSE2: cost of 2 {{.*}} %V8i16 = trunc
  ; SSSE3: cost of 2 {{.*}} %V8i16 = trunc
  ; SSE42: cost of 1 {{.*}} %V8i16 = trunc
  ; AVX: cost of 1 {{.*}} %V8i16 = trunc
  ; AVX512: cost of 1 {{.*}} %V8i16 = trunc
  %V8i16 = trunc <8 x i16> undef to <8 x i8>

  ; SSE: cost of 3 {{.*}} %V16i16 = trunc
  ; AVX: cost of 4 {{.*}} %V16i16 = trunc
  ; AVX512: cost of 4 {{.*}} %V16i16 = trunc
  %V16i16 = trunc <16 x i16> undef to <16 x i8>

  ; SSE: cost of 7 {{.*}} %V32i16 = trunc
  ; AVX: cost of 9 {{.*}} %V32i16 = trunc
  ; AVX512F: cost of 9 {{.*}} %V32i16 = trunc
  ; AVX512BW: cost of 0 {{.*}} %V32i16 = trunc
  %V32i16 = trunc <32 x i16> undef to <32 x i8>

  ret i32 undef
}
