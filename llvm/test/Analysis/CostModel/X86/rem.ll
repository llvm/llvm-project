; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX1
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'srem'
define i32 @srem() {
  ; CHECK: cost of 1 {{.*}} %I64 = srem
  %I64 = srem i64 undef, undef
  ; SSE: cost of 6 {{.*}} %V2i64 = srem
  ; AVX: cost of 6 {{.*}} %V2i64 = srem
  %V2i64 = srem <2 x i64> undef, undef
  ; SSE: cost of 12 {{.*}} %V4i64 = srem
  ; AVX: cost of 12 {{.*}} %V4i64 = srem
  %V4i64 = srem <4 x i64> undef, undef
  ; SSE: cost of 24 {{.*}} %V8i64 = srem
  ; AVX: cost of 24 {{.*}} %V8i64 = srem
  %V8i64 = srem <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = srem
  %I32 = srem i32 undef, undef
  ; SSE: cost of 12 {{.*}} %V4i32 = srem
  ; AVX: cost of 12 {{.*}} %V4i32 = srem
  %V4i32 = srem <4 x i32> undef, undef
  ; SSE: cost of 24 {{.*}} %V8i32 = srem
  ; AVX: cost of 24 {{.*}} %V8i32 = srem
  %V8i32 = srem <8 x i32> undef, undef
  ; SSE: cost of 48 {{.*}} %V16i32 = srem
  ; AVX: cost of 48 {{.*}} %V16i32 = srem
  %V16i32 = srem <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = srem
  %I16 = srem i16 undef, undef
  ; SSE: cost of 24 {{.*}} %V8i16 = srem
  ; AVX: cost of 24 {{.*}} %V8i16 = srem
  %V8i16 = srem <8 x i16> undef, undef
  ; SSE: cost of 48 {{.*}} %V16i16 = srem
  ; AVX: cost of 48 {{.*}} %V16i16 = srem
  %V16i16 = srem <16 x i16> undef, undef
  ; SSE: cost of 96 {{.*}} %V32i16 = srem
  ; AVX: cost of 96 {{.*}} %V32i16 = srem
  %V32i16 = srem <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = srem
  %I8 = srem i8 undef, undef
  ; SSE: cost of 48 {{.*}} %V16i8 = srem
  ; AVX: cost of 48 {{.*}} %V16i8 = srem
  %V16i8 = srem <16 x i8> undef, undef
  ; SSE: cost of 96 {{.*}} %V32i8 = srem
  ; AVX: cost of 96 {{.*}} %V32i8 = srem
  %V32i8 = srem <32 x i8> undef, undef
  ; SSE: cost of 192 {{.*}} %V64i8 = srem
  ; AVX: cost of 192 {{.*}} %V64i8 = srem
  %V64i8 = srem <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'urem'
define i32 @urem() {
  ; CHECK: cost of 1 {{.*}} %I64 = urem
  %I64 = urem i64 undef, undef
  ; SSE: cost of 6 {{.*}} %V2i64 = urem
  ; AVX: cost of 6 {{.*}} %V2i64 = urem
  %V2i64 = urem <2 x i64> undef, undef
  ; SSE: cost of 12 {{.*}} %V4i64 = urem
  ; AVX: cost of 12 {{.*}} %V4i64 = urem
  %V4i64 = urem <4 x i64> undef, undef
  ; SSE: cost of 24 {{.*}} %V8i64 = urem
  ; AVX: cost of 24 {{.*}} %V8i64 = urem
  %V8i64 = urem <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = urem
  %I32 = urem i32 undef, undef
  ; SSE: cost of 12 {{.*}} %V4i32 = urem
  ; AVX: cost of 12 {{.*}} %V4i32 = urem
  %V4i32 = urem <4 x i32> undef, undef
  ; SSE: cost of 24 {{.*}} %V8i32 = urem
  ; AVX: cost of 24 {{.*}} %V8i32 = urem
  %V8i32 = urem <8 x i32> undef, undef
  ; SSE: cost of 48 {{.*}} %V16i32 = urem
  ; AVX: cost of 48 {{.*}} %V16i32 = urem
  %V16i32 = urem <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = urem
  %I16 = urem i16 undef, undef
  ; SSE: cost of 24 {{.*}} %V8i16 = urem
  ; AVX: cost of 24 {{.*}} %V8i16 = urem
  %V8i16 = urem <8 x i16> undef, undef
  ; SSE: cost of 48 {{.*}} %V16i16 = urem
  ; AVX: cost of 48 {{.*}} %V16i16 = urem
  %V16i16 = urem <16 x i16> undef, undef
  ; SSE: cost of 96 {{.*}} %V32i16 = urem
  ; AVX: cost of 96 {{.*}} %V32i16 = urem
  %V32i16 = urem <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = urem
  %I8 = urem i8 undef, undef
  ; SSE: cost of 48 {{.*}} %V16i8 = urem
  ; AVX: cost of 48 {{.*}} %V16i8 = urem
  %V16i8 = urem <16 x i8> undef, undef
  ; SSE: cost of 96 {{.*}} %V32i8 = urem
  ; AVX: cost of 96 {{.*}} %V32i8 = urem
  %V32i8 = urem <32 x i8> undef, undef
  ; SSE: cost of 192 {{.*}} %V64i8 = urem
  ; AVX: cost of 192 {{.*}} %V64i8 = urem
  %V64i8 = urem <64 x i8> undef, undef

  ret i32 undef
}
