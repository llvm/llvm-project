; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+ssse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSSE3
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+sse4.2 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

;
; Verify the cost model for broadcast shuffles.
;

; CHECK-LABEL: 'test_vXf64'
define void @test_vXf64(<2 x double> %src128, <4 x double> %src256, <8 x double> %src512) {
  ; SSE: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer

  ret void
}

; CHECK-LABEL: 'test_vXi64'
define void @test_vXi64(<2 x i64> %src128, <4 x i64> %src256, <8 x i64> %src512) {
  ; SSE: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer

  ret void
}

; CHECK-LABEL: 'test_vXf32'
define void @test_vXf32(<2 x float> %src64, <4 x float> %src128, <8 x float> %src256, <16 x float> %src512) {
  ; SSE: cost of 1 {{.*}} %V64 = shufflevector
  ; AVX: cost of 1 {{.*}} %V64 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V64 = shufflevector
  %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer

  ret void
}

; CHECK-LABEL: 'test_vXi32'
define void @test_vXi32(<2 x i32> %src64, <4 x i32> %src128, <8 x i32> %src256, <16 x i32> %src512) {
  ; SSE: cost of 1 {{.*}} %V64 = shufflevector
  ; AVX: cost of 1 {{.*}} %V64 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V64 = shufflevector
  %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer

  ; SSE: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer

  ret void
}

; CHECK-LABEL: 'test_vXi16'
define void @test_vXi16(<8 x i16> %src128, <16 x i16> %src256, <32 x i16> %src512) {
  ; SSE2: cost of 2 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer

  ; SSE2: cost of 2 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 3 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer

  ; SSE2: cost of 2 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 3 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512F: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512BW: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer

  ret void
}

; CHECK-LABEL: 'test_vXi8'
define void @test_vXi8(<16 x i8> %src128, <32 x i8> %src256, <64 x i8> %src512) {
  ; SSE2: cost of 3 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer

  ; SSE2: cost of 3 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer

  ; SSE2: cost of 3 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 2 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512F: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512BW: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer

  ret void
}
