; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSSE3
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512dq | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512DQ

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'add'
define i32 @add(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = add
  %I64 = add i64 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V2I64 = add
  ; SSE42: cost of 1 {{.*}} %V2I64 = add
  ; AVX: cost of 1 {{.*}} %V2I64 = add
  ; AVX2: cost of 1 {{.*}} %V2I64 = add
  ; AVX512: cost of 1 {{.*}} %V2I64 = add
  %V2I64 = add <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V4I64 = add
  ; SSE42: cost of 2 {{.*}} %V4I64 = add
  ; AVX: cost of 4 {{.*}} %V4I64 = add
  ; AVX2: cost of 1 {{.*}} %V4I64 = add
  ; AVX512: cost of 1 {{.*}} %V4I64 = add
  %V4I64 = add <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V8I64 = add
  ; SSE42: cost of 4 {{.*}} %V8I64 = add
  ; AVX: cost of 8 {{.*}} %V8I64 = add
  ; AVX2: cost of 2 {{.*}} %V8I64 = add
  ; AVX512: cost of 1 {{.*}} %V8I64 = add
  %V8I64 = add <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = add
  %I32 = add i32 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V4I32 = add
  ; SSE42: cost of 1 {{.*}} %V4I32 = add
  ; AVX: cost of 1 {{.*}} %V4I32 = add
  ; AVX2: cost of 1 {{.*}} %V4I32 = add
  ; AVX512: cost of 1 {{.*}} %V4I32 = add
  %V4I32 = add <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V8I32 = add
  ; SSE42: cost of 2 {{.*}} %V8I32 = add
  ; AVX: cost of 4 {{.*}} %V8I32 = add
  ; AVX2: cost of 1 {{.*}} %V8I32 = add
  ; AVX512: cost of 1 {{.*}} %V8I32 = add
  %V8I32 = add <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V16I32 = add
  ; SSE42: cost of 4 {{.*}} %V16I32 = add
  ; AVX: cost of 8 {{.*}} %V16I32 = add
  ; AVX2: cost of 2 {{.*}} %V16I32 = add
  ; AVX512: cost of 1 {{.*}} %V16I32 = add
  %V16I32 = add <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = add
  %I16 = add i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = add
  ; SSE42: cost of 1 {{.*}} %V8I16 = add
  ; AVX: cost of 1 {{.*}} %V8I16 = add
  ; AVX2: cost of 1 {{.*}} %V8I16 = add
  ; AVX512: cost of 1 {{.*}} %V8I16 = add
  %V8I16 = add <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = add
  ; SSE42: cost of 2 {{.*}} %V16I16 = add
  ; AVX: cost of 4 {{.*}} %V16I16 = add
  ; AVX2: cost of 1 {{.*}} %V16I16 = add
  ; AVX512: cost of 1 {{.*}} %V16I16 = add
  %V16I16 = add <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = add
  ; SSE42: cost of 4 {{.*}} %V32I16 = add
  ; AVX: cost of 8 {{.*}} %V32I16 = add
  ; AVX2: cost of 2 {{.*}} %V32I16 = add
  ; AVX512F: cost of 2 {{.*}} %V32I16 = add
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = add
  %V32I16 = add <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = add
  %I8 = add i8 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V16I8 = add
  ; SSE42: cost of 1 {{.*}} %V16I8 = add
  ; AVX: cost of 1 {{.*}} %V16I8 = add
  ; AVX2: cost of 1 {{.*}} %V16I8 = add
  ; AVX512: cost of 1 {{.*}} %V16I8 = add
  %V16I8 = add <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V32I8 = add
  ; SSE42: cost of 2 {{.*}} %V32I8 = add
  ; AVX: cost of 4 {{.*}} %V32I8 = add
  ; AVX2: cost of 1 {{.*}} %V32I8 = add
  ; AVX512: cost of 1 {{.*}} %V32I8 = add
  %V32I8 = add <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V64I8 = add
  ; SSE42: cost of 4 {{.*}} %V64I8 = add
  ; AVX: cost of 8 {{.*}} %V64I8 = add
  ; AVX2: cost of 2 {{.*}} %V64I8 = add
  ; AVX512F: cost of 2 {{.*}} %V64I8 = add
  ; AVX512BW: cost of 1 {{.*}} %V64I8 = add
  %V64I8 = add <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'sub'
define i32 @sub(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = sub
  %I64 = sub i64 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V2I64 = sub
  ; SSE42: cost of 1 {{.*}} %V2I64 = sub
  ; AVX: cost of 1 {{.*}} %V2I64 = sub
  ; AVX2: cost of 1 {{.*}} %V2I64 = sub
  ; AVX512: cost of 1 {{.*}} %V2I64 = sub
  %V2I64 = sub <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V4I64 = sub
  ; SSE42: cost of 2 {{.*}} %V4I64 = sub
  ; AVX: cost of 4 {{.*}} %V4I64 = sub
  ; AVX2: cost of 1 {{.*}} %V4I64 = sub
  ; AVX512: cost of 1 {{.*}} %V4I64 = sub
  %V4I64 = sub <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V8I64 = sub
  ; SSE42: cost of 4 {{.*}} %V8I64 = sub
  ; AVX: cost of 8 {{.*}} %V8I64 = sub
  ; AVX2: cost of 2 {{.*}} %V8I64 = sub
  ; AVX512: cost of 1 {{.*}} %V8I64 = sub
  %V8I64 = sub <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = sub
  %I32 = sub i32 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V4I32 = sub
  ; SSE42: cost of 1 {{.*}} %V4I32 = sub
  ; AVX: cost of 1 {{.*}} %V4I32 = sub
  ; AVX2: cost of 1 {{.*}} %V4I32 = sub
  ; AVX512: cost of 1 {{.*}} %V4I32 = sub
  %V4I32 = sub <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V8I32 = sub
  ; SSE42: cost of 2 {{.*}} %V8I32 = sub
  ; AVX: cost of 4 {{.*}} %V8I32 = sub
  ; AVX2: cost of 1 {{.*}} %V8I32 = sub
  ; AVX512: cost of 1 {{.*}} %V8I32 = sub
  %V8I32 = sub <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V16I32 = sub
  ; SSE42: cost of 4 {{.*}} %V16I32 = sub
  ; AVX: cost of 8 {{.*}} %V16I32 = sub
  ; AVX2: cost of 2 {{.*}} %V16I32 = sub
  ; AVX512: cost of 1 {{.*}} %V16I32 = sub
  %V16I32 = sub <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = sub
  %I16 = sub i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = sub
  ; SSE42: cost of 1 {{.*}} %V8I16 = sub
  ; AVX: cost of 1 {{.*}} %V8I16 = sub
  ; AVX2: cost of 1 {{.*}} %V8I16 = sub
  ; AVX512: cost of 1 {{.*}} %V8I16 = sub
  %V8I16 = sub <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = sub
  ; SSE42: cost of 2 {{.*}} %V16I16 = sub
  ; AVX: cost of 4 {{.*}} %V16I16 = sub
  ; AVX2: cost of 1 {{.*}} %V16I16 = sub
  ; AVX512: cost of 1 {{.*}} %V16I16 = sub
  %V16I16 = sub <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = sub
  ; SSE42: cost of 4 {{.*}} %V32I16 = sub
  ; AVX: cost of 8 {{.*}} %V32I16 = sub
  ; AVX2: cost of 2 {{.*}} %V32I16 = sub
  ; AVX512F: cost of 2 {{.*}} %V32I16 = sub
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = sub
  %V32I16 = sub <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = sub
  %I8 = sub i8 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V16I8 = sub
  ; SSE42: cost of 1 {{.*}} %V16I8 = sub
  ; AVX: cost of 1 {{.*}} %V16I8 = sub
  ; AVX2: cost of 1 {{.*}} %V16I8 = sub
  ; AVX512: cost of 1 {{.*}} %V16I8 = sub
  %V16I8 = sub <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V32I8 = sub
  ; SSE42: cost of 2 {{.*}} %V32I8 = sub
  ; AVX: cost of 4 {{.*}} %V32I8 = sub
  ; AVX2: cost of 1 {{.*}} %V32I8 = sub
  ; AVX512: cost of 1 {{.*}} %V32I8 = sub
  %V32I8 = sub <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V64I8 = sub
  ; SSE42: cost of 4 {{.*}} %V64I8 = sub
  ; AVX: cost of 8 {{.*}} %V64I8 = sub
  ; AVX2: cost of 2 {{.*}} %V64I8 = sub
  ; AVX512F: cost of 2 {{.*}} %V64I8 = sub
  ; AVX512BW: cost of 1 {{.*}} %V64I8 = sub
  %V64I8 = sub <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'or'
define i32 @or(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = or
  %I64 = or i64 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V2I64 = or
  ; SSE42: cost of 1 {{.*}} %V2I64 = or
  ; AVX: cost of 1 {{.*}} %V2I64 = or
  ; AVX2: cost of 1 {{.*}} %V2I64 = or
  ; AVX512: cost of 1 {{.*}} %V2I64 = or
  %V2I64 = or <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V4I64 = or
  ; SSE42: cost of 2 {{.*}} %V4I64 = or
  ; AVX: cost of 1 {{.*}} %V4I64 = or
  ; AVX2: cost of 1 {{.*}} %V4I64 = or
  ; AVX512: cost of 1 {{.*}} %V4I64 = or
  %V4I64 = or <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V8I64 = or
  ; SSE42: cost of 4 {{.*}} %V8I64 = or
  ; AVX: cost of 2 {{.*}} %V8I64 = or
  ; AVX2: cost of 2 {{.*}} %V8I64 = or
  ; AVX512: cost of 1 {{.*}} %V8I64 = or
  %V8I64 = or <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = or
  %I32 = or i32 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V4I32 = or
  ; SSE42: cost of 1 {{.*}} %V4I32 = or
  ; AVX: cost of 1 {{.*}} %V4I32 = or
  ; AVX2: cost of 1 {{.*}} %V4I32 = or
  ; AVX512: cost of 1 {{.*}} %V4I32 = or
  %V4I32 = or <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V8I32 = or
  ; SSE42: cost of 2 {{.*}} %V8I32 = or
  ; AVX: cost of 1 {{.*}} %V8I32 = or
  ; AVX2: cost of 1 {{.*}} %V8I32 = or
  ; AVX512: cost of 1 {{.*}} %V8I32 = or
  %V8I32 = or <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V16I32 = or
  ; SSE42: cost of 4 {{.*}} %V16I32 = or
  ; AVX: cost of 2 {{.*}} %V16I32 = or
  ; AVX2: cost of 2 {{.*}} %V16I32 = or
  ; AVX512: cost of 1 {{.*}} %V16I32 = or
  %V16I32 = or <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = or
  %I16 = or i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = or
  ; SSE42: cost of 1 {{.*}} %V8I16 = or
  ; AVX: cost of 1 {{.*}} %V8I16 = or
  ; AVX2: cost of 1 {{.*}} %V8I16 = or
  ; AVX512: cost of 1 {{.*}} %V8I16 = or
  %V8I16 = or <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = or
  ; SSE42: cost of 2 {{.*}} %V16I16 = or
  ; AVX: cost of 1 {{.*}} %V16I16 = or
  ; AVX2: cost of 1 {{.*}} %V16I16 = or
  ; AVX512: cost of 1 {{.*}} %V16I16 = or
  %V16I16 = or <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = or
  ; SSE42: cost of 4 {{.*}} %V32I16 = or
  ; AVX: cost of 2 {{.*}} %V32I16 = or
  ; AVX2: cost of 2 {{.*}} %V32I16 = or
  ; AVX512F: cost of 2 {{.*}} %V32I16 = or
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = or
  %V32I16 = or <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = or
  %I8 = or i8 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V16I8 = or
  ; SSE42: cost of 1 {{.*}} %V16I8 = or
  ; AVX: cost of 1 {{.*}} %V16I8 = or
  ; AVX2: cost of 1 {{.*}} %V16I8 = or
  ; AVX512: cost of 1 {{.*}} %V16I8 = or
  %V16I8 = or <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V32I8 = or
  ; SSE42: cost of 2 {{.*}} %V32I8 = or
  ; AVX: cost of 1 {{.*}} %V32I8 = or
  ; AVX2: cost of 1 {{.*}} %V32I8 = or
  ; AVX512: cost of 1 {{.*}} %V32I8 = or
  %V32I8 = or <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V64I8 = or
  ; SSE42: cost of 4 {{.*}} %V64I8 = or
  ; AVX: cost of 2 {{.*}} %V64I8 = or
  ; AVX2: cost of 2 {{.*}} %V64I8 = or
  ; AVX512F: cost of 2 {{.*}} %V64I8 = or
  ; AVX512BW: cost of 1 {{.*}} %V64I8 = or
  %V64I8 = or <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'xor'
define i32 @xor(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = xor
  %I64 = xor i64 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V2I64 = xor
  ; SSE42: cost of 1 {{.*}} %V2I64 = xor
  ; AVX: cost of 1 {{.*}} %V2I64 = xor
  ; AVX2: cost of 1 {{.*}} %V2I64 = xor
  ; AVX512: cost of 1 {{.*}} %V2I64 = xor
  %V2I64 = xor <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V4I64 = xor
  ; SSE42: cost of 2 {{.*}} %V4I64 = xor
  ; AVX: cost of 1 {{.*}} %V4I64 = xor
  ; AVX2: cost of 1 {{.*}} %V4I64 = xor
  ; AVX512: cost of 1 {{.*}} %V4I64 = xor
  %V4I64 = xor <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V8I64 = xor
  ; SSE42: cost of 4 {{.*}} %V8I64 = xor
  ; AVX: cost of 2 {{.*}} %V8I64 = xor
  ; AVX2: cost of 2 {{.*}} %V8I64 = xor
  ; AVX512: cost of 1 {{.*}} %V8I64 = xor
  %V8I64 = xor <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = xor
  %I32 = xor i32 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V4I32 = xor
  ; SSE42: cost of 1 {{.*}} %V4I32 = xor
  ; AVX: cost of 1 {{.*}} %V4I32 = xor
  ; AVX2: cost of 1 {{.*}} %V4I32 = xor
  ; AVX512: cost of 1 {{.*}} %V4I32 = xor
  %V4I32 = xor <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V8I32 = xor
  ; SSE42: cost of 2 {{.*}} %V8I32 = xor
  ; AVX: cost of 1 {{.*}} %V8I32 = xor
  ; AVX2: cost of 1 {{.*}} %V8I32 = xor
  ; AVX512: cost of 1 {{.*}} %V8I32 = xor
  %V8I32 = xor <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V16I32 = xor
  ; SSE42: cost of 4 {{.*}} %V16I32 = xor
  ; AVX: cost of 2 {{.*}} %V16I32 = xor
  ; AVX2: cost of 2 {{.*}} %V16I32 = xor
  ; AVX512: cost of 1 {{.*}} %V16I32 = xor
  %V16I32 = xor <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = xor
  %I16 = xor i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = xor
  ; SSE42: cost of 1 {{.*}} %V8I16 = xor
  ; AVX: cost of 1 {{.*}} %V8I16 = xor
  ; AVX2: cost of 1 {{.*}} %V8I16 = xor
  ; AVX512: cost of 1 {{.*}} %V8I16 = xor
  %V8I16 = xor <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = xor
  ; SSE42: cost of 2 {{.*}} %V16I16 = xor
  ; AVX: cost of 1 {{.*}} %V16I16 = xor
  ; AVX2: cost of 1 {{.*}} %V16I16 = xor
  ; AVX512: cost of 1 {{.*}} %V16I16 = xor
  %V16I16 = xor <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = xor
  ; SSE42: cost of 4 {{.*}} %V32I16 = xor
  ; AVX: cost of 2 {{.*}} %V32I16 = xor
  ; AVX2: cost of 2 {{.*}} %V32I16 = xor
  ; AVX512F: cost of 2 {{.*}} %V32I16 = xor
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = xor
  %V32I16 = xor <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = xor
  %I8 = xor i8 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V16I8 = xor
  ; SSE42: cost of 1 {{.*}} %V16I8 = xor
  ; AVX: cost of 1 {{.*}} %V16I8 = xor
  ; AVX2: cost of 1 {{.*}} %V16I8 = xor
  ; AVX512: cost of 1 {{.*}} %V16I8 = xor
  %V16I8 = xor <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V32I8 = xor
  ; SSE42: cost of 2 {{.*}} %V32I8 = xor
  ; AVX: cost of 1 {{.*}} %V32I8 = xor
  ; AVX2: cost of 1 {{.*}} %V32I8 = xor
  ; AVX512: cost of 1 {{.*}} %V32I8 = xor
  %V32I8 = xor <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V64I8 = xor
  ; SSE42: cost of 4 {{.*}} %V64I8 = xor
  ; AVX: cost of 2 {{.*}} %V64I8 = xor
  ; AVX2: cost of 2 {{.*}} %V64I8 = xor
  ; AVX512F: cost of 2 {{.*}} %V64I8 = xor
  ; AVX512BW: cost of 1 {{.*}} %V64I8 = xor
  %V64I8 = xor <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'and'
define i32 @and(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = and
  %I64 = and i64 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V2I64 = and
  ; SSE42: cost of 1 {{.*}} %V2I64 = and
  ; AVX: cost of 1 {{.*}} %V2I64 = and
  ; AVX2: cost of 1 {{.*}} %V2I64 = and
  ; AVX512: cost of 1 {{.*}} %V2I64 = and
  %V2I64 = and <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V4I64 = and
  ; SSE42: cost of 2 {{.*}} %V4I64 = and
  ; AVX: cost of 1 {{.*}} %V4I64 = and
  ; AVX2: cost of 1 {{.*}} %V4I64 = and
  ; AVX512: cost of 1 {{.*}} %V4I64 = and
  %V4I64 = and <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V8I64 = and
  ; SSE42: cost of 4 {{.*}} %V8I64 = and
  ; AVX: cost of 2 {{.*}} %V8I64 = and
  ; AVX2: cost of 2 {{.*}} %V8I64 = and
  ; AVX512: cost of 1 {{.*}} %V8I64 = and
  %V8I64 = and <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = and
  %I32 = and i32 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V4I32 = and
  ; SSE42: cost of 1 {{.*}} %V4I32 = and
  ; AVX: cost of 1 {{.*}} %V4I32 = and
  ; AVX2: cost of 1 {{.*}} %V4I32 = and
  ; AVX512: cost of 1 {{.*}} %V4I32 = and
  %V4I32 = and <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V8I32 = and
  ; SSE42: cost of 2 {{.*}} %V8I32 = and
  ; AVX: cost of 1 {{.*}} %V8I32 = and
  ; AVX2: cost of 1 {{.*}} %V8I32 = and
  ; AVX512: cost of 1 {{.*}} %V8I32 = and
  %V8I32 = and <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V16I32 = and
  ; SSE42: cost of 4 {{.*}} %V16I32 = and
  ; AVX: cost of 2 {{.*}} %V16I32 = and
  ; AVX2: cost of 2 {{.*}} %V16I32 = and
  ; AVX512: cost of 1 {{.*}} %V16I32 = and
  %V16I32 = and <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = and
  %I16 = and i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = and
  ; SSE42: cost of 1 {{.*}} %V8I16 = and
  ; AVX: cost of 1 {{.*}} %V8I16 = and
  ; AVX2: cost of 1 {{.*}} %V8I16 = and
  ; AVX512: cost of 1 {{.*}} %V8I16 = and
  %V8I16 = and <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = and
  ; SSE42: cost of 2 {{.*}} %V16I16 = and
  ; AVX: cost of 1 {{.*}} %V16I16 = and
  ; AVX2: cost of 1 {{.*}} %V16I16 = and
  ; AVX512: cost of 1 {{.*}} %V16I16 = and
  %V16I16 = and <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = and
  ; SSE42: cost of 4 {{.*}} %V32I16 = and
  ; AVX: cost of 2 {{.*}} %V32I16 = and
  ; AVX2: cost of 2 {{.*}} %V32I16 = and
  ; AVX512F: cost of 2 {{.*}} %V32I16 = and
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = and
  %V32I16 = and <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = and
  %I8 = and i8 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V16I8 = and
  ; SSE42: cost of 1 {{.*}} %V16I8 = and
  ; AVX: cost of 1 {{.*}} %V16I8 = and
  ; AVX2: cost of 1 {{.*}} %V16I8 = and
  ; AVX512: cost of 1 {{.*}} %V16I8 = and
  %V16I8 = and <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V32I8 = and
  ; SSE42: cost of 2 {{.*}} %V32I8 = and
  ; AVX: cost of 1 {{.*}} %V32I8 = and
  ; AVX2: cost of 1 {{.*}} %V32I8 = and
  ; AVX512: cost of 1 {{.*}} %V32I8 = and
  %V32I8 = and <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V64I8 = and
  ; SSE42: cost of 4 {{.*}} %V64I8 = and
  ; AVX: cost of 2 {{.*}} %V64I8 = and
  ; AVX2: cost of 2 {{.*}} %V64I8 = and
  ; AVX512F: cost of 2 {{.*}} %V64I8 = and
  ; AVX512BW: cost of 1 {{.*}} %V64I8 = and
  %V64I8 = and <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'mul'
define i32 @mul(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} %I64 = mul
  %I64 = mul i64 undef, undef
  ; SSSE3: cost of 8 {{.*}} %V2I64 = mul
  ; SSE42: cost of 8 {{.*}} %V2I64 = mul
  ; AVX: cost of 8 {{.*}} %V2I64 = mul
  ; AVX2: cost of 8 {{.*}} %V2I64 = mul
  ; AVX512F: cost of 8 {{.*}} %V2I64 = mul
  ; AVX512BW: cost of 8 {{.*}} %V2I64 = mul
  ; AVX512DQ: cost of 1 {{.*}} %V2I64 = mul
  %V2I64 = mul <2 x i64> undef, undef
  ; SSSE3: cost of 16 {{.*}} %V4I64 = mul
  ; SSE42: cost of 16 {{.*}} %V4I64 = mul
  ; AVX: cost of 18 {{.*}} %V4I64 = mul
  ; AVX2: cost of 8 {{.*}} %V4I64 = mul
  ; AVX512F: cost of 8 {{.*}} %V4I64 = mul
  ; AVX512BW: cost of 8 {{.*}} %V4I64 = mul
  ; AVX512DQ: cost of 1 {{.*}} %V4I64 = mul
  %V4I64 = mul <4 x i64> undef, undef
  ; SSSE3: cost of 32 {{.*}} %V8I64 = mul
  ; SSE42: cost of 32 {{.*}} %V8I64 = mul
  ; AVX: cost of 36 {{.*}} %V8I64 = mul
  ; AVX2: cost of 16 {{.*}} %V8I64 = mul
  ; AVX512F: cost of 8 {{.*}} %V8I64 = mul
  ; AVX512BW: cost of 8 {{.*}} %V8I64 = mul
  ; AVX512DQ: cost of 1 {{.*}} %V8I64 = mul
  %V8I64 = mul <8 x i64> undef, undef

  ; CHECK: cost of 1 {{.*}} %I32 = mul
  %I32 = mul i32 undef, undef
  ; SSSE3: cost of 6 {{.*}} %V4I32 = mul
  ; SSE42: cost of 1 {{.*}} %V4I32 = mul
  ; AVX: cost of 1 {{.*}} %V4I32 = mul
  ; AVX2: cost of 1 {{.*}} %V4I32 = mul
  ; AVX512: cost of 1 {{.*}} %V4I32 = mul
  %V4I32 = mul <4 x i32> undef, undef
  ; SSSE3: cost of 12 {{.*}} %V8I32 = mul
  ; SSE42: cost of 2 {{.*}} %V8I32 = mul
  ; AVX: cost of 4 {{.*}} %V8I32 = mul
  ; AVX2: cost of 1 {{.*}} %V8I32 = mul
  ; AVX512: cost of 1 {{.*}} %V8I32 = mul
  %V8I32 = mul <8 x i32> undef, undef
  ; SSSE3: cost of 24 {{.*}} %V16I32 = mul
  ; SSE42: cost of 4 {{.*}} %V16I32 = mul
  ; AVX: cost of 8 {{.*}} %V16I32 = mul
  ; AVX2: cost of 2 {{.*}} %V16I32 = mul
  ; AVX512: cost of 1 {{.*}} %V16I32 = mul
  %V16I32 = mul <16 x i32> undef, undef

  ; CHECK: cost of 1 {{.*}} %I16 = mul
  %I16 = mul i16 undef, undef
  ; SSSE3: cost of 1 {{.*}} %V8I16 = mul
  ; SSE42: cost of 1 {{.*}} %V8I16 = mul
  ; AVX: cost of 1 {{.*}} %V8I16 = mul
  ; AVX2: cost of 1 {{.*}} %V8I16 = mul
  ; AVX512: cost of 1 {{.*}} %V8I16 = mul
  %V8I16 = mul <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %V16I16 = mul
  ; SSE42: cost of 2 {{.*}} %V16I16 = mul
  ; AVX: cost of 4 {{.*}} %V16I16 = mul
  ; AVX2: cost of 1 {{.*}} %V16I16 = mul
  ; AVX512: cost of 1 {{.*}} %V16I16 = mul
  %V16I16 = mul <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %V32I16 = mul
  ; SSE42: cost of 4 {{.*}} %V32I16 = mul
  ; AVX: cost of 8 {{.*}} %V32I16 = mul
  ; AVX2: cost of 2 {{.*}} %V32I16 = mul
  ; AVX512F: cost of 2 {{.*}} %V32I16 = mul
  ; AVX512BW: cost of 1 {{.*}} %V32I16 = mul
  %V32I16 = mul <32 x i16> undef, undef

  ; CHECK: cost of 1 {{.*}} %I8 = mul
  %I8 = mul i8 undef, undef
  ; SSSE3: cost of 12 {{.*}} %V16I8 = mul
  ; SSE42: cost of 12 {{.*}} %V16I8 = mul
  ; AVX: cost of 12 {{.*}} %V16I8 = mul
  ; AVX2: cost of 7 {{.*}} %V16I8 = mul
  ; AVX512F: cost of 5 {{.*}} %V16I8 = mul
  ; AVX512BW: cost of 4 {{.*}} %V16I8 = mul
  %V16I8 = mul <16 x i8> undef, undef
  ; SSSE3: cost of 24 {{.*}} %V32I8 = mul
  ; SSE42: cost of 24 {{.*}} %V32I8 = mul
  ; AVX: cost of 26 {{.*}} %V32I8 = mul
  ; AVX2: cost of 17 {{.*}} %V32I8 = mul
  ; AVX512F: cost of 13 {{.*}} %V32I8 = mul
  ; AVX512BW: cost of 4 {{.*}} %V32I8 = mul
  %V32I8 = mul <32 x i8> undef, undef
  ; SSSE3: cost of 48 {{.*}} %V64I8 = mul
  ; SSE42: cost of 48 {{.*}} %V64I8 = mul
  ; AVX: cost of 52 {{.*}} %V64I8 = mul
  ; AVX2: cost of 34 {{.*}} %V64I8 = mul
  ; AVX512F: cost of 26 {{.*}} %V64I8 = mul
  ; AVX512BW: cost of 11 {{.*}} %V64I8 = mul
  %V64I8 = mul <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'mul_2i32'
define void @mul_2i32() {
  ; A <2 x i32> gets expanded to a <2 x i64> vector.
  ; A <2 x i64> vector multiply is implemented using
  ; 3 PMULUDQ and 2 PADDS and 4 shifts.
  ; SSSE3: cost of 8 {{.*}} %A0 = mul
  ; SSE42: cost of 8 {{.*}} %A0 = mul
  ; AVX: cost of 8 {{.*}} %A0 = mul
  ; AVX2: cost of 8 {{.*}} %A0 = mul
  ; AVX512F: cost of 8 {{.*}} %A0 = mul
  ; AVX512BW: cost of 8 {{.*}} %A0 = mul
  ; AVX512DQ: cost of 1 {{.*}} %A0 = mul
  %A0 = mul <2 x i32> undef, undef

  ret void
}
