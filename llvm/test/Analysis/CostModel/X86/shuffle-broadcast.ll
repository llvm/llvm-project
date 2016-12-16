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
  ; SSE: Unknown cost {{.*}} %V128 = shufflevector
  ; AVX: Unknown cost {{.*}} %V128 = shufflevector
  ; AVX512: Unknown cost {{.*}} %V128 = shufflevector
  %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer

  ; SSE: Unknown cost {{.*}} %V256 = shufflevector
  ; AVX: Unknown cost {{.*}} %V256 = shufflevector
  ; AVX512: Unknown cost {{.*}} %V256 = shufflevector
  %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer

  ; SSE: Unknown cost {{.*}} %V512 = shufflevector
  ; AVX: Unknown cost {{.*}} %V512 = shufflevector
  ; AVX512: Unknown cost {{.*}} %V512 = shufflevector
  %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer

  ret void
}
