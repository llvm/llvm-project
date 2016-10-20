; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX1
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'sdiv'
define i32 @sdiv() {
  ; CHECK: cost of 1 {{.*}} %I64 = sdiv
  %I64 = sdiv i64 undef, undef
  ; SSE: cost of 40 {{.*}} %V2i64 = sdiv
  ; AVX1: cost of 40 {{.*}} %V2i64 = sdiv
  ; AVX2: cost of 40 {{.*}} %V2i64 = sdiv
  ; AVX512: cost of 40 {{.*}} %V2i64 = sdiv
  %V2i64 = sdiv <2 x i64> undef, undef
  ; SSE: cost of 80 {{.*}} %V4i64 = sdiv
  ; AVX1: cost of 12 {{.*}} %V4i64 = sdiv
  ; AVX2: cost of 80 {{.*}} %V4i64 = sdiv
  ; AVX512: cost of 80 {{.*}} %V4i64 = sdiv
  %V4i64 = sdiv <4 x i64> undef, undef
  ; SSE: cost of 160 {{.*}} %V8i64 = sdiv
  ; AVX1: cost of 24 {{.*}} %V8i64 = sdiv
  ; AVX2: cost of 160 {{.*}} %V8i64 = sdiv
  ; AVX512: cost of 24 {{.*}} %V8i64 = sdiv
  %V8i64 = sdiv <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = sdiv
  %I32 = sdiv i32 undef, undef
  ; SSE: cost of 80 {{.*}} %V4i32 = sdiv
  ; AVX1: cost of 80 {{.*}} %V4i32 = sdiv
  ; AVX2: cost of 80 {{.*}} %V4i32 = sdiv
  ; AVX512: cost of 80 {{.*}} %V4i32 = sdiv
  %V4i32 = sdiv <4 x i32> undef, undef
  ; SSE: cost of 160 {{.*}} %V8i32 = sdiv
  ; AVX1: cost of 24 {{.*}} %V8i32 = sdiv
  ; AVX2: cost of 160 {{.*}} %V8i32 = sdiv
  ; AVX512: cost of 160 {{.*}} %V8i32 = sdiv
  %V8i32 = sdiv <8 x i32> undef, undef
  ; SSE: cost of 320 {{.*}} %V16i32 = sdiv
  ; AVX1: cost of 48 {{.*}} %V16i32 = sdiv
  ; AVX2: cost of 320 {{.*}} %V16i32 = sdiv
  ; AVX512: cost of 48 {{.*}} %V16i32 = sdiv
  %V16i32 = sdiv <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = sdiv
  %I16 = sdiv i16 undef, undef
  ; SSE: cost of 160 {{.*}} %V8i16 = sdiv
  ; AVX1: cost of 160 {{.*}} %V8i16 = sdiv
  ; AVX2: cost of 160 {{.*}} %V8i16 = sdiv
  ; AVX512: cost of 160 {{.*}} %V8i16 = sdiv
  %V8i16 = sdiv <8 x i16> undef, undef
  ; SSE: cost of 320 {{.*}} %V16i16 = sdiv
  ; AVX1: cost of 48 {{.*}} %V16i16 = sdiv
  ; AVX2: cost of 320 {{.*}} %V16i16 = sdiv
  ; AVX512: cost of 320 {{.*}} %V16i16 = sdiv
  %V16i16 = sdiv <16 x i16> undef, undef
  ; SSE: cost of 640 {{.*}} %V32i16 = sdiv
  ; AVX1: cost of 96 {{.*}} %V32i16 = sdiv
  ; AVX2: cost of 640 {{.*}} %V32i16 = sdiv
  ; AVX512F: cost of 640 {{.*}} %V32i16 = sdiv
  ; AVX512BW: cost of 96 {{.*}} %V32i16 = sdiv
  %V32i16 = sdiv <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = sdiv
  %I8 = sdiv i8 undef, undef
  ; SSE: cost of 320 {{.*}} %V16i8 = sdiv
  ; AVX1: cost of 320 {{.*}} %V16i8 = sdiv
  ; AVX2: cost of 320 {{.*}} %V16i8 = sdiv
  ; AVX512: cost of 320 {{.*}} %V16i8 = sdiv
  %V16i8 = sdiv <16 x i8> undef, undef
  ; SSE: cost of 640 {{.*}} %V32i8 = sdiv
  ; AVX1: cost of 96 {{.*}} %V32i8 = sdiv
  ; AVX2: cost of 640 {{.*}} %V32i8 = sdiv
  ; AVX512: cost of 640 {{.*}} %V32i8 = sdiv
  %V32i8 = sdiv <32 x i8> undef, undef
  ; SSE: cost of 1280 {{.*}} %V64i8 = sdiv
  ; AVX1: cost of 192 {{.*}} %V64i8 = sdiv
  ; AVX2: cost of 1280 {{.*}} %V64i8 = sdiv
  ; AVX512F: cost of 1280 {{.*}} %V64i8 = sdiv
  ; AVX512BW: cost of 192 {{.*}} %V64i8 = sdiv
  %V64i8 = sdiv <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'udiv'
define i32 @udiv() {
  ; CHECK: cost of 1 {{.*}} %I64 = udiv
  %I64 = udiv i64 undef, undef
  ; SSE: cost of 40 {{.*}} %V2i64 = udiv
  ; AVX1: cost of 40 {{.*}} %V2i64 = udiv
  ; AVX2: cost of 40 {{.*}} %V2i64 = udiv
  ; AVX512: cost of 40 {{.*}} %V2i64 = udiv
  %V2i64 = udiv <2 x i64> undef, undef
  ; SSE: cost of 80 {{.*}} %V4i64 = udiv
  ; AVX1: cost of 12 {{.*}} %V4i64 = udiv
  ; AVX2: cost of 80 {{.*}} %V4i64 = udiv
  ; AVX512: cost of 80 {{.*}} %V4i64 = udiv
  %V4i64 = udiv <4 x i64> undef, undef
  ; SSE: cost of 160 {{.*}} %V8i64 = udiv
  ; AVX1: cost of 24 {{.*}} %V8i64 = udiv
  ; AVX2: cost of 160 {{.*}} %V8i64 = udiv
  ; AVX512: cost of 24 {{.*}} %V8i64 = udiv
  %V8i64 = udiv <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = udiv
  %I32 = udiv i32 undef, undef
  ; SSE: cost of 80 {{.*}} %V4i32 = udiv
  ; AVX1: cost of 80 {{.*}} %V4i32 = udiv
  ; AVX2: cost of 80 {{.*}} %V4i32 = udiv
  ; AVX512: cost of 80 {{.*}} %V4i32 = udiv
  %V4i32 = udiv <4 x i32> undef, undef
  ; SSE: cost of 160 {{.*}} %V8i32 = udiv
  ; AVX1: cost of 24 {{.*}} %V8i32 = udiv
  ; AVX2: cost of 160 {{.*}} %V8i32 = udiv
  ; AVX512: cost of 160 {{.*}} %V8i32 = udiv
  %V8i32 = udiv <8 x i32> undef, undef
  ; SSE: cost of 320 {{.*}} %V16i32 = udiv
  ; AVX1: cost of 48 {{.*}} %V16i32 = udiv
  ; AVX2: cost of 320 {{.*}} %V16i32 = udiv
  ; AVX512: cost of 48 {{.*}} %V16i32 = udiv
  %V16i32 = udiv <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = udiv
  %I16 = udiv i16 undef, undef
  ; SSE: cost of 160 {{.*}} %V8i16 = udiv
  ; AVX1: cost of 160 {{.*}} %V8i16 = udiv
  ; AVX2: cost of 160 {{.*}} %V8i16 = udiv
  ; AVX512: cost of 160 {{.*}} %V8i16 = udiv
  %V8i16 = udiv <8 x i16> undef, undef
  ; SSE: cost of 320 {{.*}} %V16i16 = udiv
  ; AVX1: cost of 48 {{.*}} %V16i16 = udiv
  ; AVX2: cost of 320 {{.*}} %V16i16 = udiv
  ; AVX512: cost of 320 {{.*}} %V16i16 = udiv
  %V16i16 = udiv <16 x i16> undef, undef
  ; SSE: cost of 640 {{.*}} %V32i16 = udiv
  ; AVX1: cost of 96 {{.*}} %V32i16 = udiv
  ; AVX2: cost of 640 {{.*}} %V32i16 = udiv
  ; AVX512F: cost of 640 {{.*}} %V32i16 = udiv
  ; AVX512BW: cost of 96 {{.*}} %V32i16 = udiv
  %V32i16 = udiv <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = udiv
  %I8 = udiv i8 undef, undef
  ; SSE: cost of 320 {{.*}} %V16i8 = udiv
  ; AVX1: cost of 320 {{.*}} %V16i8 = udiv
  ; AVX2: cost of 320 {{.*}} %V16i8 = udiv
  ; AVX512: cost of 320 {{.*}} %V16i8 = udiv
  %V16i8 = udiv <16 x i8> undef, undef
  ; SSE: cost of 640 {{.*}} %V32i8 = udiv
  ; AVX1: cost of 96 {{.*}} %V32i8 = udiv
  ; AVX2: cost of 640 {{.*}} %V32i8 = udiv
  ; AVX512: cost of 640 {{.*}} %V32i8 = udiv
  %V32i8 = udiv <32 x i8> undef, undef
  ; SSE: cost of 1280 {{.*}} %V64i8 = udiv
  ; AVX1: cost of 192 {{.*}} %V64i8 = udiv
  ; AVX2: cost of 1280 {{.*}} %V64i8 = udiv
  ; AVX512F: cost of 1280 {{.*}} %V64i8 = udiv
  ; AVX512BW: cost of 192 {{.*}} %V64i8 = udiv
  %V64i8 = udiv <64 x i8> undef, undef

  ret i32 undef
}
