; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2
; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx,+fma | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2,+fma | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -enable-no-nans-fp-math  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'fadd'
define i32 @fadd(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = fadd
  ; SSE42: cost of 2 {{.*}} %F32 = fadd
  ; AVX: cost of 2 {{.*}} %F32 = fadd
  ; AVX2: cost of 2 {{.*}} %F32 = fadd
  ; AVX512: cost of 2 {{.*}} %F32 = fadd
  %F32 = fadd float undef, undef
  ; SSE2: cost of 2 {{.*}} %V4F32 = fadd
  ; SSE42: cost of 2 {{.*}} %V4F32 = fadd
  ; AVX: cost of 2 {{.*}} %V4F32 = fadd
  ; AVX2: cost of 2 {{.*}} %V4F32 = fadd
  ; AVX512: cost of 2 {{.*}} %V4F32 = fadd
  %V4F32 = fadd <4 x float> undef, undef
  ; SSE2: cost of 4 {{.*}} %V8F32 = fadd
  ; SSE42: cost of 4 {{.*}} %V8F32 = fadd
  ; AVX: cost of 2 {{.*}} %V8F32 = fadd
  ; AVX2: cost of 2 {{.*}} %V8F32 = fadd
  ; AVX512: cost of 2 {{.*}} %V8F32 = fadd
  %V8F32 = fadd <8 x float> undef, undef
  ; SSE2: cost of 8 {{.*}} %V16F32 = fadd
  ; SSE42: cost of 8 {{.*}} %V16F32 = fadd
  ; AVX: cost of 4 {{.*}} %V16F32 = fadd
  ; AVX2: cost of 4 {{.*}} %V16F32 = fadd
  ; AVX512: cost of 2 {{.*}} %V16F32 = fadd
  %V16F32 = fadd <16 x float> undef, undef

  ; SSE2: cost of 2 {{.*}} %F64 = fadd
  ; SSE42: cost of 2 {{.*}} %F64 = fadd
  ; AVX: cost of 2 {{.*}} %F64 = fadd
  ; AVX2: cost of 2 {{.*}} %F64 = fadd
  ; AVX512: cost of 2 {{.*}} %F64 = fadd
  %F64 = fadd double undef, undef
  ; SSE2: cost of 2 {{.*}} %V2F64 = fadd
  ; SSE42: cost of 2 {{.*}} %V2F64 = fadd
  ; AVX: cost of 2 {{.*}} %V2F64 = fadd
  ; AVX2: cost of 2 {{.*}} %V2F64 = fadd
  ; AVX512: cost of 2 {{.*}} %V2F64 = fadd
  %V2F64 = fadd <2 x double> undef, undef
  ; SSE2: cost of 4 {{.*}} %V4F64 = fadd
  ; SSE42: cost of 4 {{.*}} %V4F64 = fadd
  ; AVX: cost of 2 {{.*}} %V4F64 = fadd
  ; AVX2: cost of 2 {{.*}} %V4F64 = fadd
  ; AVX512: cost of 2 {{.*}} %V4F64 = fadd
  %V4F64 = fadd <4 x double> undef, undef
  ; SSE2: cost of 8 {{.*}} %V8F64 = fadd
  ; SSE42: cost of 8 {{.*}} %V8F64 = fadd
  ; AVX: cost of 4 {{.*}} %V8F64 = fadd
  ; AVX2: cost of 4 {{.*}} %V8F64 = fadd
  ; AVX512: cost of 2 {{.*}} %V8F64 = fadd
  %V8F64 = fadd <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fsub'
define i32 @fsub(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = fsub
  ; SSE42: cost of 2 {{.*}} %F32 = fsub
  ; AVX: cost of 2 {{.*}} %F32 = fsub
  ; AVX2: cost of 2 {{.*}} %F32 = fsub
  ; AVX512: cost of 2 {{.*}} %F32 = fsub
  %F32 = fsub float undef, undef
  ; SSE2: cost of 2 {{.*}} %V4F32 = fsub
  ; SSE42: cost of 2 {{.*}} %V4F32 = fsub
  ; AVX: cost of 2 {{.*}} %V4F32 = fsub
  ; AVX2: cost of 2 {{.*}} %V4F32 = fsub
  ; AVX512: cost of 2 {{.*}} %V4F32 = fsub
  %V4F32 = fsub <4 x float> undef, undef
  ; SSE2: cost of 4 {{.*}} %V8F32 = fsub
  ; SSE42: cost of 4 {{.*}} %V8F32 = fsub
  ; AVX: cost of 2 {{.*}} %V8F32 = fsub
  ; AVX2: cost of 2 {{.*}} %V8F32 = fsub
  ; AVX512: cost of 2 {{.*}} %V8F32 = fsub
  %V8F32 = fsub <8 x float> undef, undef
  ; SSE2: cost of 8 {{.*}} %V16F32 = fsub
  ; SSE42: cost of 8 {{.*}} %V16F32 = fsub
  ; AVX: cost of 4 {{.*}} %V16F32 = fsub
  ; AVX2: cost of 4 {{.*}} %V16F32 = fsub
  ; AVX512: cost of 2 {{.*}} %V16F32 = fsub
  %V16F32 = fsub <16 x float> undef, undef

  ; SSE2: cost of 2 {{.*}} %F64 = fsub
  ; SSE42: cost of 2 {{.*}} %F64 = fsub
  ; AVX: cost of 2 {{.*}} %F64 = fsub
  ; AVX2: cost of 2 {{.*}} %F64 = fsub
  ; AVX512: cost of 2 {{.*}} %F64 = fsub
  %F64 = fsub double undef, undef
  ; SSE2: cost of 2 {{.*}} %V2F64 = fsub
  ; SSE42: cost of 2 {{.*}} %V2F64 = fsub
  ; AVX: cost of 2 {{.*}} %V2F64 = fsub
  ; AVX2: cost of 2 {{.*}} %V2F64 = fsub
  ; AVX512: cost of 2 {{.*}} %V2F64 = fsub
  %V2F64 = fsub <2 x double> undef, undef
  ; SSE2: cost of 4 {{.*}} %V4F64 = fsub
  ; SSE42: cost of 4 {{.*}} %V4F64 = fsub
  ; AVX: cost of 2 {{.*}} %V4F64 = fsub
  ; AVX2: cost of 2 {{.*}} %V4F64 = fsub
  ; AVX512: cost of 2 {{.*}} %V4F64 = fsub
  %V4F64 = fsub <4 x double> undef, undef
  ; SSE2: cost of 8 {{.*}} %V8F64 = fsub
  ; SSE42: cost of 8 {{.*}} %V8F64 = fsub
  ; AVX: cost of 4 {{.*}} %V8F64 = fsub
  ; AVX2: cost of 4 {{.*}} %V8F64 = fsub
  ; AVX512: cost of 2 {{.*}} %V8F64 = fsub
  %V8F64 = fsub <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fmul'
define i32 @fmul(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = fmul
  ; SSE42: cost of 2 {{.*}} %F32 = fmul
  ; AVX: cost of 2 {{.*}} %F32 = fmul
  ; AVX2: cost of 2 {{.*}} %F32 = fmul
  ; AVX512: cost of 2 {{.*}} %F32 = fmul
  %F32 = fmul float undef, undef
  ; SSE2: cost of 2 {{.*}} %V4F32 = fmul
  ; SSE42: cost of 2 {{.*}} %V4F32 = fmul
  ; AVX: cost of 2 {{.*}} %V4F32 = fmul
  ; AVX2: cost of 2 {{.*}} %V4F32 = fmul
  ; AVX512: cost of 2 {{.*}} %V4F32 = fmul
  %V4F32 = fmul <4 x float> undef, undef
  ; SSE2: cost of 4 {{.*}} %V8F32 = fmul
  ; SSE42: cost of 4 {{.*}} %V8F32 = fmul
  ; AVX: cost of 2 {{.*}} %V8F32 = fmul
  ; AVX2: cost of 2 {{.*}} %V8F32 = fmul
  ; AVX512: cost of 2 {{.*}} %V8F32 = fmul
  %V8F32 = fmul <8 x float> undef, undef
  ; SSE2: cost of 8 {{.*}} %V16F32 = fmul
  ; SSE42: cost of 8 {{.*}} %V16F32 = fmul
  ; AVX: cost of 4 {{.*}} %V16F32 = fmul
  ; AVX2: cost of 4 {{.*}} %V16F32 = fmul
  ; AVX512: cost of 2 {{.*}} %V16F32 = fmul
  %V16F32 = fmul <16 x float> undef, undef

  ; SSE2: cost of 2 {{.*}} %F64 = fmul
  ; SSE42: cost of 2 {{.*}} %F64 = fmul
  ; AVX: cost of 2 {{.*}} %F64 = fmul
  ; AVX2: cost of 2 {{.*}} %F64 = fmul
  ; AVX512: cost of 2 {{.*}} %F64 = fmul
  %F64 = fmul double undef, undef
  ; SSE2: cost of 2 {{.*}} %V2F64 = fmul
  ; SSE42: cost of 2 {{.*}} %V2F64 = fmul
  ; AVX: cost of 2 {{.*}} %V2F64 = fmul
  ; AVX2: cost of 2 {{.*}} %V2F64 = fmul
  ; AVX512: cost of 2 {{.*}} %V2F64 = fmul
  %V2F64 = fmul <2 x double> undef, undef
  ; SSE2: cost of 4 {{.*}} %V4F64 = fmul
  ; SSE42: cost of 4 {{.*}} %V4F64 = fmul
  ; AVX: cost of 2 {{.*}} %V4F64 = fmul
  ; AVX2: cost of 2 {{.*}} %V4F64 = fmul
  ; AVX512: cost of 2 {{.*}} %V4F64 = fmul
  %V4F64 = fmul <4 x double> undef, undef
  ; SSE2: cost of 8 {{.*}} %V8F64 = fmul
  ; SSE42: cost of 8 {{.*}} %V8F64 = fmul
  ; AVX: cost of 4 {{.*}} %V8F64 = fmul
  ; AVX2: cost of 4 {{.*}} %V8F64 = fmul
  ; AVX512: cost of 2 {{.*}} %V8F64 = fmul
  %V8F64 = fmul <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fdiv'
define i32 @fdiv(i32 %arg) {
  ; SSE2: cost of 23 {{.*}} %F32 = fdiv
  ; SSE42: cost of 14 {{.*}} %F32 = fdiv
  ; AVX: cost of 14 {{.*}} %F32 = fdiv
  ; AVX2: cost of 7 {{.*}} %F32 = fdiv
  ; AVX512: cost of 7 {{.*}} %F32 = fdiv
  %F32 = fdiv float undef, undef
  ; SSE2: cost of 39 {{.*}} %V4F32 = fdiv
  ; SSE42: cost of 14 {{.*}} %V4F32 = fdiv
  ; AVX: cost of 14 {{.*}} %V4F32 = fdiv
  ; AVX2: cost of 7 {{.*}} %V4F32 = fdiv
  ; AVX512: cost of 7 {{.*}} %V4F32 = fdiv
  %V4F32 = fdiv <4 x float> undef, undef
  ; SSE2: cost of 78 {{.*}} %V8F32 = fdiv
  ; SSE42: cost of 28 {{.*}} %V8F32 = fdiv
  ; AVX: cost of 28 {{.*}} %V8F32 = fdiv
  ; AVX2: cost of 14 {{.*}} %V8F32 = fdiv
  ; AVX512: cost of 14 {{.*}} %V8F32 = fdiv
  %V8F32 = fdiv <8 x float> undef, undef
  ; SSE2: cost of 156 {{.*}} %V16F32 = fdiv
  ; SSE42: cost of 56 {{.*}} %V16F32 = fdiv
  ; AVX: cost of 56 {{.*}} %V16F32 = fdiv
  ; AVX2: cost of 28 {{.*}} %V16F32 = fdiv
  ; AVX512: cost of 2 {{.*}} %V16F32 = fdiv
  %V16F32 = fdiv <16 x float> undef, undef

  ; SSE2: cost of 38 {{.*}} %F64 = fdiv
  ; SSE42: cost of 22 {{.*}} %F64 = fdiv
  ; AVX: cost of 22 {{.*}} %F64 = fdiv
  ; AVX2: cost of 14 {{.*}} %F64 = fdiv
  ; AVX512: cost of 14 {{.*}} %F64 = fdiv
  %F64 = fdiv double undef, undef
  ; SSE2: cost of 69 {{.*}} %V2F64 = fdiv
  ; SSE42: cost of 22 {{.*}} %V2F64 = fdiv
  ; AVX: cost of 22 {{.*}} %V2F64 = fdiv
  ; AVX2: cost of 14 {{.*}} %V2F64 = fdiv
  ; AVX512: cost of 14 {{.*}} %V2F64 = fdiv
  %V2F64 = fdiv <2 x double> undef, undef
  ; SSE2: cost of 138 {{.*}} %V4F64 = fdiv
  ; SSE42: cost of 44 {{.*}} %V4F64 = fdiv
  ; AVX: cost of 44 {{.*}} %V4F64 = fdiv
  ; AVX2: cost of 28 {{.*}} %V4F64 = fdiv
  ; AVX512: cost of 28 {{.*}} %V4F64 = fdiv
  %V4F64 = fdiv <4 x double> undef, undef
  ; SSE2: cost of 276 {{.*}} %V8F64 = fdiv
  ; SSE42: cost of 88 {{.*}} %V8F64 = fdiv
  ; AVX: cost of 88 {{.*}} %V8F64 = fdiv
  ; AVX2: cost of 56 {{.*}} %V8F64 = fdiv
  ; AVX512: cost of 2 {{.*}} %V8F64 = fdiv
  %V8F64 = fdiv <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'frem'
define i32 @frem(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = frem
  ; SSE42: cost of 2 {{.*}} %F32 = frem
  ; AVX: cost of 2 {{.*}} %F32 = frem
  ; AVX2: cost of 2 {{.*}} %F32 = frem
  ; AVX512: cost of 2 {{.*}} %F32 = frem
  %F32 = frem float undef, undef
  ; SSE2: cost of 14 {{.*}} %V4F32 = frem
  ; SSE42: cost of 14 {{.*}} %V4F32 = frem
  ; AVX: cost of 14 {{.*}} %V4F32 = frem
  ; AVX2: cost of 14 {{.*}} %V4F32 = frem
  ; AVX512: cost of 14 {{.*}} %V4F32 = frem
  %V4F32 = frem <4 x float> undef, undef
  ; SSE2: cost of 28 {{.*}} %V8F32 = frem
  ; SSE42: cost of 28 {{.*}} %V8F32 = frem
  ; AVX: cost of 30 {{.*}} %V8F32 = frem
  ; AVX2: cost of 30 {{.*}} %V8F32 = frem
  ; AVX512: cost of 30 {{.*}} %V8F32 = frem
  %V8F32 = frem <8 x float> undef, undef
  ; SSE2: cost of 56 {{.*}} %V16F32 = frem
  ; SSE42: cost of 56 {{.*}} %V16F32 = frem
  ; AVX: cost of 60 {{.*}} %V16F32 = frem
  ; AVX2: cost of 60 {{.*}} %V16F32 = frem
  ; AVX512: cost of 62 {{.*}} %V16F32 = frem
  %V16F32 = frem <16 x float> undef, undef

  ; SSE2: cost of 2 {{.*}} %F64 = frem
  ; SSE42: cost of 2 {{.*}} %F64 = frem
  ; AVX: cost of 2 {{.*}} %F64 = frem
  ; AVX2: cost of 2 {{.*}} %F64 = frem
  ; AVX512: cost of 2 {{.*}} %F64 = frem
  %F64 = frem double undef, undef
  ; SSE2: cost of 6 {{.*}} %V2F64 = frem
  ; SSE42: cost of 6 {{.*}} %V2F64 = frem
  ; AVX: cost of 6 {{.*}} %V2F64 = frem
  ; AVX2: cost of 6 {{.*}} %V2F64 = frem
  ; AVX512: cost of 6 {{.*}} %V2F64 = frem
  %V2F64 = frem <2 x double> undef, undef
  ; SSE2: cost of 12 {{.*}} %V4F64 = frem
  ; SSE42: cost of 12 {{.*}} %V4F64 = frem
  ; AVX: cost of 14 {{.*}} %V4F64 = frem
  ; AVX2: cost of 14 {{.*}} %V4F64 = frem
  ; AVX512: cost of 14 {{.*}} %V4F64 = frem
  %V4F64 = frem <4 x double> undef, undef
  ; SSE2: cost of 24 {{.*}} %V8F64 = frem
  ; SSE42: cost of 24 {{.*}} %V8F64 = frem
  ; AVX: cost of 28 {{.*}} %V8F64 = frem
  ; AVX2: cost of 28 {{.*}} %V8F64 = frem
  ; AVX512: cost of 30 {{.*}} %V8F64 = frem
  %V8F64 = frem <8 x double> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'fsqrt'
define i32 @fsqrt(i32 %arg) {
  ; SSE2: cost of 28 {{.*}} %F32 = call float @llvm.sqrt.f32
  ; SSE42: cost of 18 {{.*}} %F32 = call float @llvm.sqrt.f32
  ; AVX: cost of 14 {{.*}} %F32 = call float @llvm.sqrt.f32
  ; AVX2: cost of 7 {{.*}} %F32 = call float @llvm.sqrt.f32
  ; AVX512: cost of 7 {{.*}} %F32 = call float @llvm.sqrt.f32
  %F32 = call float @llvm.sqrt.f32(float undef)
  ; SSE2: cost of 56 {{.*}} %V4F32 = call <4 x float> @llvm.sqrt.v4f32
  ; SSE42: cost of 18 {{.*}} %V4F32 = call <4 x float> @llvm.sqrt.v4f32
  ; AVX: cost of 14 {{.*}} %V4F32 = call <4 x float> @llvm.sqrt.v4f32
  ; AVX2: cost of 7 {{.*}} %V4F32 = call <4 x float> @llvm.sqrt.v4f32
  ; AVX512: cost of 7 {{.*}} %V4F32 = call <4 x float> @llvm.sqrt.v4f32
  %V4F32 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> undef)
  ; SSE2: cost of 112 {{.*}} %V8F32 = call <8 x float> @llvm.sqrt.v8f32
  ; SSE42: cost of 36 {{.*}} %V8F32 = call <8 x float> @llvm.sqrt.v8f32
  ; AVX: cost of 28 {{.*}} %V8F32 = call <8 x float> @llvm.sqrt.v8f32
  ; AVX2: cost of 14 {{.*}} %V8F32 = call <8 x float> @llvm.sqrt.v8f32
  ; AVX512: cost of 14 {{.*}} %V8F32 = call <8 x float> @llvm.sqrt.v8f32
  %V8F32 = call <8 x float> @llvm.sqrt.v8f32(<8 x float> undef)
  ; SSE2: cost of 224 {{.*}} %V16F32 = call <16 x float> @llvm.sqrt.v16f32
  ; SSE42: cost of 72 {{.*}} %V16F32 = call <16 x float> @llvm.sqrt.v16f32
  ; AVX: cost of 56 {{.*}} %V16F32 = call <16 x float> @llvm.sqrt.v16f32
  ; AVX2: cost of 28 {{.*}} %V16F32 = call <16 x float> @llvm.sqrt.v16f32
  ; AVX512: cost of 1 {{.*}} %V16F32 = call <16 x float> @llvm.sqrt.v16f32
  %V16F32 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> undef)

  ; SSE2: cost of 32 {{.*}} %F64 = call double @llvm.sqrt.f64
  ; SSE42: cost of 32 {{.*}} %F64 = call double @llvm.sqrt.f64
  ; AVX: cost of 21 {{.*}} %F64 = call double @llvm.sqrt.f64
  ; AVX2: cost of 14 {{.*}} %F64 = call double @llvm.sqrt.f64
  ; AVX512: cost of 14 {{.*}} %F64 = call double @llvm.sqrt.f64
  %F64 = call double @llvm.sqrt.f64(double undef)
  ; SSE2: cost of 32 {{.*}} %V2F64 = call <2 x double> @llvm.sqrt.v2f64
  ; SSE42: cost of 32 {{.*}} %V2F64 = call <2 x double> @llvm.sqrt.v2f64
  ; AVX: cost of 21 {{.*}} %V2F64 = call <2 x double> @llvm.sqrt.v2f64
  ; AVX2: cost of 14 {{.*}} %V2F64 = call <2 x double> @llvm.sqrt.v2f64
  ; AVX512: cost of 14 {{.*}} %V2F64 = call <2 x double> @llvm.sqrt.v2f64
  %V2F64 = call <2 x double> @llvm.sqrt.v2f64(<2 x double> undef)
  ; SSE2: cost of 64 {{.*}} %V4F64 = call <4 x double> @llvm.sqrt.v4f64
  ; SSE42: cost of 64 {{.*}} %V4F64 = call <4 x double> @llvm.sqrt.v4f64
  ; AVX: cost of 43 {{.*}} %V4F64 = call <4 x double> @llvm.sqrt.v4f64
  ; AVX2: cost of 28 {{.*}} %V4F64 = call <4 x double> @llvm.sqrt.v4f64
  ; AVX512: cost of 28 {{.*}} %V4F64 = call <4 x double> @llvm.sqrt.v4f64
  %V4F64 = call <4 x double> @llvm.sqrt.v4f64(<4 x double> undef)
  ; SSE2: cost of 128 {{.*}} %V8F64 = call <8 x double> @llvm.sqrt.v8f64
  ; SSE42: cost of 128 {{.*}} %V8F64 = call <8 x double> @llvm.sqrt.v8f64
  ; AVX: cost of 86 {{.*}} %V8F64 = call <8 x double> @llvm.sqrt.v8f64
  ; AVX2: cost of 56 {{.*}} %V8F64 = call <8 x double> @llvm.sqrt.v8f64
  ; AVX512: cost of 1 {{.*}} %V8F64 = call <8 x double> @llvm.sqrt.v8f64
  %V8F64 = call <8 x double> @llvm.sqrt.v8f64(<8 x double> undef)

  ret i32 undef
}

; CHECK-LABEL: 'fabs'
define i32 @fabs(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = call float @llvm.fabs.f32
  ; SSE42: cost of 2 {{.*}} %F32 = call float @llvm.fabs.f32
  ; AVX: cost of 2 {{.*}} %F32 = call float @llvm.fabs.f32
  ; AVX2: cost of 2 {{.*}} %F32 = call float @llvm.fabs.f32
  ; AVX512: cost of 2 {{.*}} %F32 = call float @llvm.fabs.f32
  %F32 = call float @llvm.fabs.f32(float undef)
  ; SSE2: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.fabs.v4f32
  ; SSE42: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.fabs.v4f32
  ; AVX: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.fabs.v4f32
  ; AVX2: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.fabs.v4f32
  ; AVX512: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.fabs.v4f32
  %V4F32 = call <4 x float> @llvm.fabs.v4f32(<4 x float> undef)
  ; SSE2: cost of 4 {{.*}} %V8F32 = call <8 x float> @llvm.fabs.v8f32
  ; SSE42: cost of 4 {{.*}} %V8F32 = call <8 x float> @llvm.fabs.v8f32
  ; AVX: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.fabs.v8f32
  ; AVX2: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.fabs.v8f32
  ; AVX512: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.fabs.v8f32
  %V8F32 = call <8 x float> @llvm.fabs.v8f32(<8 x float> undef)
  ; SSE2: cost of 8 {{.*}} %V16F32 = call <16 x float> @llvm.fabs.v16f32
  ; SSE42: cost of 8 {{.*}} %V16F32 = call <16 x float> @llvm.fabs.v16f32
  ; AVX: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.fabs.v16f32
  ; AVX2: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.fabs.v16f32
  ; AVX512: cost of 2 {{.*}} %V16F32 = call <16 x float> @llvm.fabs.v16f32
  %V16F32 = call <16 x float> @llvm.fabs.v16f32(<16 x float> undef)

  ; SSE2: cost of 2 {{.*}} %F64 = call double @llvm.fabs.f64
  ; SSE42: cost of 2 {{.*}} %F64 = call double @llvm.fabs.f64
  ; AVX: cost of 2 {{.*}} %F64 = call double @llvm.fabs.f64
  ; AVX2: cost of 2 {{.*}} %F64 = call double @llvm.fabs.f64
  ; AVX512: cost of 2 {{.*}} %F64 = call double @llvm.fabs.f64
  %F64 = call double @llvm.fabs.f64(double undef)
  ; SSE2: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.fabs.v2f64
  ; SSE42: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.fabs.v2f64
  ; AVX: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.fabs.v2f64
  ; AVX2: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.fabs.v2f64
  ; AVX512: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.fabs.v2f64
  %V2F64 = call <2 x double> @llvm.fabs.v2f64(<2 x double> undef)
  ; SSE2: cost of 4 {{.*}} %V4F64 = call <4 x double> @llvm.fabs.v4f64
  ; SSE42: cost of 4 {{.*}} %V4F64 = call <4 x double> @llvm.fabs.v4f64
  ; AVX: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.fabs.v4f64
  ; AVX2: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.fabs.v4f64
  ; AVX512: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.fabs.v4f64
  %V4F64 = call <4 x double> @llvm.fabs.v4f64(<4 x double> undef)
  ; SSE2: cost of 8 {{.*}} %V8F64 = call <8 x double> @llvm.fabs.v8f64
  ; SSE42: cost of 8 {{.*}} %V8F64 = call <8 x double> @llvm.fabs.v8f64
  ; AVX: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.fabs.v8f64
  ; AVX2: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.fabs.v8f64
  ; AVX512: cost of 2 {{.*}} %V8F64 = call <8 x double> @llvm.fabs.v8f64
  %V8F64 = call <8 x double> @llvm.fabs.v8f64(<8 x double> undef)

  ret i32 undef
}

; CHECK-LABEL: 'fcopysign'
define i32 @fcopysign(i32 %arg) {
  ; SSE2: cost of 2 {{.*}} %F32 = call float @llvm.copysign.f32
  ; SSE42: cost of 2 {{.*}} %F32 = call float @llvm.copysign.f32
  ; AVX: cost of 2 {{.*}} %F32 = call float @llvm.copysign.f32
  ; AVX2: cost of 2 {{.*}} %F32 = call float @llvm.copysign.f32
  ; AVX512: cost of 2 {{.*}} %F32 = call float @llvm.copysign.f32
  %F32 = call float @llvm.copysign.f32(float undef, float undef)
  ; SSE2: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.copysign.v4f32
  ; SSE42: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.copysign.v4f32
  ; AVX: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.copysign.v4f32
  ; AVX2: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.copysign.v4f32
  ; AVX512: cost of 2 {{.*}} %V4F32 = call <4 x float> @llvm.copysign.v4f32
  %V4F32 = call <4 x float> @llvm.copysign.v4f32(<4 x float> undef, <4 x float> undef)
  ; SSE2: cost of 4 {{.*}} %V8F32 = call <8 x float> @llvm.copysign.v8f32
  ; SSE42: cost of 4 {{.*}} %V8F32 = call <8 x float> @llvm.copysign.v8f32
  ; AVX: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.copysign.v8f32
  ; AVX2: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.copysign.v8f32
  ; AVX512: cost of 2 {{.*}} %V8F32 = call <8 x float> @llvm.copysign.v8f32
  %V8F32 = call <8 x float> @llvm.copysign.v8f32(<8 x float> undef, <8 x float> undef)
  ; SSE2: cost of 8 {{.*}} %V16F32 = call <16 x float> @llvm.copysign.v16f32
  ; SSE42: cost of 8 {{.*}} %V16F32 = call <16 x float> @llvm.copysign.v16f32
  ; AVX: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.copysign.v16f32
  ; AVX2: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.copysign.v16f32
  ; AVX512: cost of 2 {{.*}} %V16F32 = call <16 x float> @llvm.copysign.v16f32
  %V16F32 = call <16 x float> @llvm.copysign.v16f32(<16 x float> undef, <16 x float> undef)

  ; SSE2: cost of 2 {{.*}} %F64 = call double @llvm.copysign.f64
  ; SSE42: cost of 2 {{.*}} %F64 = call double @llvm.copysign.f64
  ; AVX: cost of 2 {{.*}} %F64 = call double @llvm.copysign.f64
  ; AVX2: cost of 2 {{.*}} %F64 = call double @llvm.copysign.f64
  ; AVX512: cost of 2 {{.*}} %F64 = call double @llvm.copysign.f64
  %F64 = call double @llvm.copysign.f64(double undef, double undef)
  ; SSE2: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.copysign.v2f64
  ; SSE42: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.copysign.v2f64
  ; AVX: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.copysign.v2f64
  ; AVX2: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.copysign.v2f64
  ; AVX512: cost of 2 {{.*}} %V2F64 = call <2 x double> @llvm.copysign.v2f64
  %V2F64 = call <2 x double> @llvm.copysign.v2f64(<2 x double> undef, <2 x double> undef)
  ; SSE2: cost of 4 {{.*}} %V4F64 = call <4 x double> @llvm.copysign.v4f64
  ; SSE42: cost of 4 {{.*}} %V4F64 = call <4 x double> @llvm.copysign.v4f64
  ; AVX: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.copysign.v4f64
  ; AVX2: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.copysign.v4f64
  ; AVX512: cost of 2 {{.*}} %V4F64 = call <4 x double> @llvm.copysign.v4f64
  %V4F64 = call <4 x double> @llvm.copysign.v4f64(<4 x double> undef, <4 x double> undef)
  ; SSE2: cost of 8 {{.*}} %V8F64 = call <8 x double> @llvm.copysign.v8f64
  ; SSE42: cost of 8 {{.*}} %V8F64 = call <8 x double> @llvm.copysign.v8f64
  ; AVX: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.copysign.v8f64
  ; AVX2: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.copysign.v8f64
  ; AVX512: cost of 2 {{.*}} %V8F64 = call <8 x double> @llvm.copysign.v8f64
  %V8F64 = call <8 x double> @llvm.copysign.v8f64(<8 x double> undef, <8 x double> undef)

  ret i32 undef
}

; CHECK-LABEL: 'fma'
define i32 @fma(i32 %arg) {
  ; SSE2: cost of 10 {{.*}} %F32 = call float @llvm.fma.f32
  ; SSE42: cost of 10 {{.*}} %F32 = call float @llvm.fma.f32
  ; AVX: cost of 1 {{.*}} %F32 = call float @llvm.fma.f32
  ; AVX2: cost of 1 {{.*}} %F32 = call float @llvm.fma.f32
  ; AVX512: cost of 1 {{.*}} %F32 = call float @llvm.fma.f32
  %F32 = call float @llvm.fma.f32(float undef, float undef, float undef)
  ; SSE2: cost of 43 {{.*}} %V4F32 = call <4 x float> @llvm.fma.v4f32
  ; SSE42: cost of 43 {{.*}} %V4F32 = call <4 x float> @llvm.fma.v4f32
  ; AVX: cost of 1 {{.*}} %V4F32 = call <4 x float> @llvm.fma.v4f32
  ; AVX2: cost of 1 {{.*}} %V4F32 = call <4 x float> @llvm.fma.v4f32
  ; AVX512: cost of 1 {{.*}} %V4F32 = call <4 x float> @llvm.fma.v4f32
  %V4F32 = call <4 x float> @llvm.fma.v4f32(<4 x float> undef, <4 x float> undef, <4 x float> undef)
  ; SSE2: cost of 86 {{.*}} %V8F32 = call <8 x float> @llvm.fma.v8f32
  ; SSE42: cost of 86 {{.*}} %V8F32 = call <8 x float> @llvm.fma.v8f32
  ; AVX: cost of 1 {{.*}} %V8F32 = call <8 x float> @llvm.fma.v8f32
  ; AVX2: cost of 1 {{.*}} %V8F32 = call <8 x float> @llvm.fma.v8f32
  ; AVX512: cost of 1 {{.*}} %V8F32 = call <8 x float> @llvm.fma.v8f32
  %V8F32 = call <8 x float> @llvm.fma.v8f32(<8 x float> undef, <8 x float> undef, <8 x float> undef)
  ; SSE2: cost of 172 {{.*}} %V16F32 = call <16 x float> @llvm.fma.v16f32
  ; SSE42: cost of 172 {{.*}} %V16F32 = call <16 x float> @llvm.fma.v16f32
  ; AVX: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.fma.v16f32
  ; AVX2: cost of 4 {{.*}} %V16F32 = call <16 x float> @llvm.fma.v16f32
  ; AVX512: cost of 1 {{.*}} %V16F32 = call <16 x float> @llvm.fma.v16f32
  %V16F32 = call <16 x float> @llvm.fma.v16f32(<16 x float> undef, <16 x float> undef, <16 x float> undef)

  ; SSE2: cost of 10 {{.*}} %F64 = call double @llvm.fma.f64
  ; SSE42: cost of 10 {{.*}} %F64 = call double @llvm.fma.f64
  ; AVX: cost of 1 {{.*}} %F64 = call double @llvm.fma.f64
  ; AVX2: cost of 1 {{.*}} %F64 = call double @llvm.fma.f64
  ; AVX512: cost of 1 {{.*}} %F64 = call double @llvm.fma.f64
  %F64 = call double @llvm.fma.f64(double undef, double undef, double undef)
  ; SSE2: cost of 21 {{.*}} %V2F64 = call <2 x double> @llvm.fma.v2f64
  ; SSE42: cost of 21 {{.*}} %V2F64 = call <2 x double> @llvm.fma.v2f64
  ; AVX: cost of 1 {{.*}} %V2F64 = call <2 x double> @llvm.fma.v2f64
  ; AVX2: cost of 1 {{.*}} %V2F64 = call <2 x double> @llvm.fma.v2f64
  ; AVX512: cost of 1 {{.*}} %V2F64 = call <2 x double> @llvm.fma.v2f64
  %V2F64 = call <2 x double> @llvm.fma.v2f64(<2 x double> undef, <2 x double> undef, <2 x double> undef)
  ; SSE2: cost of 42 {{.*}} %V4F64 = call <4 x double> @llvm.fma.v4f64
  ; SSE42: cost of 42 {{.*}} %V4F64 = call <4 x double> @llvm.fma.v4f64
  ; AVX: cost of 1 {{.*}} %V4F64 = call <4 x double> @llvm.fma.v4f64
  ; AVX2: cost of 1 {{.*}} %V4F64 = call <4 x double> @llvm.fma.v4f64
  ; AVX512: cost of 1 {{.*}} %V4F64 = call <4 x double> @llvm.fma.v4f64
  %V4F64 = call <4 x double> @llvm.fma.v4f64(<4 x double> undef, <4 x double> undef, <4 x double> undef)
  ; SSE2: cost of 84 {{.*}} %V8F64 = call <8 x double> @llvm.fma.v8f64
  ; SSE42: cost of 84 {{.*}} %V8F64 = call <8 x double> @llvm.fma.v8f64
  ; AVX: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.fma.v8f64
  ; AVX2: cost of 4 {{.*}} %V8F64 = call <8 x double> @llvm.fma.v8f64
  ; AVX512: cost of 1 {{.*}} %V8F64 = call <8 x double> @llvm.fma.v8f64
  %V8F64 = call <8 x double> @llvm.fma.v8f64(<8 x double> undef, <8 x double> undef, <8 x double> undef)

  ret i32 undef
}

declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>)
declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)

declare double @llvm.sqrt.f64(double)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <8 x double> @llvm.sqrt.v8f64(<8 x double>)

declare float @llvm.fabs.f32(float)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>)
declare <16 x float> @llvm.fabs.v16f32(<16 x float>)

declare double @llvm.fabs.f64(double)
declare <2 x double> @llvm.fabs.v2f64(<2 x double>)
declare <4 x double> @llvm.fabs.v4f64(<4 x double>)
declare <8 x double> @llvm.fabs.v8f64(<8 x double>)

declare float @llvm.copysign.f32(float, float)
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.copysign.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.copysign.v16f32(<16 x float>, <16 x float>)

declare double @llvm.copysign.f64(double, double)
declare <2 x double> @llvm.copysign.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.copysign.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.copysign.v8f64(<8 x double>, <8 x double>)

declare float @llvm.fma.f32(float, float, float)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)
declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>)

declare double @llvm.fma.f64(double, double, double)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
declare <8 x double> @llvm.fma.v8f64(<8 x double>, <8 x double>, <8 x double>)
