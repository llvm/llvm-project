; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE2
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+ssse3 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSSE3
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+sse4.2 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=SSE42
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+xop | FileCheck %s -check-prefix=CHECK -check-prefix=XOP
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX1
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX2
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW
; RUN: opt < %s -cost-model -analyze -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vbmi | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512VBMI

;
; Verify the cost model for 1 src shuffles
;

; CHECK-LABEL: 'test_vXf64'
define void @test_vXf64(<2 x double> %src128, <4 x double> %src256, <8 x double> %src512, <16 x double> %src1024) {

  ; SSE2: cost of 1 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> <i32 1, i32 1>

  ; SSE2: cost of 2 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 2 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 2 {{.*}} %V256 = shufflevector
  ; XOP: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 3 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> <i32 3, i32 3, i32 1, i32 0>

  ; SSE2: cost of 12 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 12 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 12 {{.*}} %V512 = shufflevector
  ; XOP: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 6 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> <i32 7, i32 6, i32 6, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 56 {{.*}} %V1024 = shufflevector
  ; SSSE3: cost of 56 {{.*}} %V1024 = shufflevector
  ; SSE42: cost of 56 {{.*}} %V1024 = shufflevector
  ; XOP: cost of 48 {{.*}} %V1024 = shufflevector
  ; AVX1: cost of 48 {{.*}} %V1024 = shufflevector
  ; AVX2: cost of 36 {{.*}} %V1024 = shufflevector
  ; AVX512: cost of 2 {{.*}} %V1024 = shufflevector
  %V1024 = shufflevector <16 x double> %src1024, <16 x double> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 13, i32 11, i32 10, i32 8, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ret void
}

; CHECK-LABEL: 'test_vXi64'
define void @test_vXi64(<2 x i64> %src128, <4 x i64> %src256, <8 x i64> %src512) {

  ; SSE2: cost of 1 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> <i32 1, i32 1>

  ; SSE2: cost of 2 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 2 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 2 {{.*}} %V256 = shufflevector
  ; XOP: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 3 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> <i32 3, i32 3, i32 1, i32 0>

  ; SSE2: cost of 12 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 12 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 12 {{.*}} %V512 = shufflevector
  ; XOP: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 6 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> <i32 7, i32 6, i32 6, i32 4, i32 3, i32 2, i32 1, i32 0>

  ret void
}

; CHECK-LABEL: 'test_vXf32'
define void @test_vXf32(<4 x float> %src128, <8 x float> %src256, <16 x float> %src512) {

  ; SSE2: cost of 1 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 1, i32 0>

  ; SSE2: cost of 4 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 4 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 4 {{.*}} %V256 = shufflevector
  ; XOP: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> <i32 7, i32 6, i32 6, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 24 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 24 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 24 {{.*}} %V512 = shufflevector
  ; XOP: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 6 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 13, i32 11, i32 10, i32 8, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ret void
}

; CHECK-LABEL: 'test_vXi32'
define void @test_vXi32(<4 x i32> %src128, <8 x i32> %src256, <16 x i32> %src512, <32 x i32> %src1024) {

  ; SSE2: cost of 1 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> <i32 3, i32 3, i32 1, i32 0>

  ; SSE2: cost of 4 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 4 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 4 {{.*}} %V256 = shufflevector
  ; XOP: cost of 2 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 5, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 24 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 24 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 24 {{.*}} %V512 = shufflevector
  ; XOP: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 6 {{.*}} %V512 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 13, i32 10, i32 9, i32 8, i32 8, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 112 {{.*}} %V1024 = shufflevector
  ; SSSE3: cost of 112 {{.*}} %V1024 = shufflevector
  ; SSE42: cost of 112 {{.*}} %V1024 = shufflevector
  ; XOP: cost of 48 {{.*}} %V1024 = shufflevector
  ; AVX1: cost of 48 {{.*}} %V1024 = shufflevector
  ; AVX2: cost of 36 {{.*}} %V1024 = shufflevector
  ; AVX512: cost of 2 {{.*}} %V1024 = shufflevector
  %V1024 = shufflevector <32 x i32> %src1024, <32 x i32> undef, <32 x i32> <i32 31, i32 30, i32 20, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 11, i32 9, i32 8, i32 7, i32 11, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret void
}

; CHECK-LABEL: 'test_vXi16'
define void @test_vXi16(<8 x i16> %src128, <16 x i16> %src256, <32 x i16> %src512, <64 x i16> %src1024) {

  ; SSE2: cost of 5 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512F: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512BW: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512VBMI: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 6, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 16 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 6 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 6 {{.*}} %V256 = shufflevector
  ; XOP: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX512F: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX512BW cost of 1 {{.*}} %V256 = shufflevector
  ; AVX512VBMI cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 13, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 96 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 36 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 36 {{.*}} %V512 = shufflevector
  ; XOP: cost of 18 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 30 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 14 {{.*}} %V512 = shufflevector
  ; AVX512F: cost of 14 {{.*}} %V512 = shufflevector
  ; AVX512BW: cost of 1 {{.*}} %V512 = shufflevector
  ; AVX512VBMI: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> <i32 31, i32 30, i32 20, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 11, i32 9, i32 8, i32 7, i32 11, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 448 {{.*}} %V1024 = shufflevector
  ; SSSE3: cost of 168 {{.*}} %V1024 = shufflevector
  ; SSE42: cost of 168 {{.*}} %V1024 = shufflevector
  ; XOP: cost of 108 {{.*}} %V1024 = shufflevector
  ; AVX1: cost of 180 {{.*}} %V1024 = shufflevector
  ; AVX2: cost of 84 {{.*}} %V1024 = shufflevector
  ; AVX512F: cost of 84 {{.*}} %V1024 = shufflevector
  ; AVX512BW: cost of 2 {{.*}} %V1024 = shufflevector
  ; AVX512VBMI: cost of 2 {{.*}} %V1024 = shufflevector
  %V1024 = shufflevector <64 x i16> %src1024, <64 x i16> undef, <64 x i32> <i32 63, i32 62, i32 61, i32 60, i32 59, i32 58, i32 57, i32 56, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 47, i32 46, i32 45, i32 44, i32 43, i32 42, i32 41, i32 40, i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 20, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret void
}

; CHECK-LABEL: 'test_vXi8'
define void @test_vXi8(<16 x i8> %src128, <32 x i8> %src256, <64 x i8> %src512) {
  ; SSE2: cost of 10 {{.*}} %V128 = shufflevector
  ; SSSE3: cost of 1 {{.*}} %V128 = shufflevector
  ; SSE42: cost of 1 {{.*}} %V128 = shufflevector
  ; XOP: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX1: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX2: cost of 1 {{.*}} %V128 = shufflevector
  ; AVX512: cost of 1 {{.*}} %V128 = shufflevector
  %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 11, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 26 {{.*}} %V256 = shufflevector
  ; SSSE3: cost of 6 {{.*}} %V256 = shufflevector
  ; SSE42: cost of 6 {{.*}} %V256 = shufflevector
  ; XOP: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX1: cost of 8 {{.*}} %V256 = shufflevector
  ; AVX2: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX512F: cost of 4 {{.*}} %V256 = shufflevector
  ; AVX512BW: cost of 3 {{.*}} %V256 = shufflevector
  ; AVX512VBMI: cost of 1 {{.*}} %V256 = shufflevector
  %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 8, i32 8, i32 7, i32 6, i32 8, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; SSE2: cost of 156 {{.*}} %V512 = shufflevector
  ; SSSE3: cost of 36 {{.*}} %V512 = shufflevector
  ; SSE42: cost of 36 {{.*}} %V512 = shufflevector
  ; XOP: cost of 18 {{.*}} %V512 = shufflevector
  ; AVX1: cost of 30 {{.*}} %V512 = shufflevector
  ; AVX2: cost of 14 {{.*}} %V512 = shufflevector
  ; AVX512F: cost of 14 {{.*}} %V512 = shufflevector
  ; AVX512BW: cost of 8 {{.*}} %V512 = shufflevector
  ; AVX512VBMI: cost of 1 {{.*}} %V512 = shufflevector
  %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> <i32 63, i32 62, i32 61, i32 60, i32 59, i32 58, i32 57, i32 56, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 47, i32 46, i32 45, i32 44, i32 43, i32 42, i32 41, i32 40, i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 20, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ret void
}
