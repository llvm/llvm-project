; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; --------------------------------------------------------------------
; Wrong mangled types
; --------------------------------------------------------------------

; CHECK: operand 0 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i64.v8i32(<4 x i64> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 0, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i64> %arg0
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v4i64_fp8__v8i32_fp8(<4 x i64> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i64.v8i32(<4 x i64> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: operand 1 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i64(<8 x i32> %arg0, <4 x i64> %arg1, <4 x float> %arg2, i32 0, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i64> %arg1
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_fp8v4i64_fp8(<8 x i32> %arg0, <4 x i64> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i64(<8 x i32> %arg0, <4 x i64> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: operand 0 must be 4, 6 or 8 element i32 vector
; CHECK:   %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i64.v8i32(<4 x i64> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK: <4 x i64> %arg0
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v4i64_fp8__v8i32_fp8(<4 x i64> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i64.v8i32(<4 x i64> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; CHECK: operand 1 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i64(<8 x i32> %arg0, <4 x i64> %arg1, <16 x float> %arg2, i32 0, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i64> %arg1
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_fp8v4i64_fp8(<8 x i32> %arg0, <4 x i64> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i64(<8 x i32> %arg0, <4 x i64> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; --------------------------------------------------------------------
; Impossible vector types
; --------------------------------------------------------------------

; CHECK: operand 0 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v5i32.v8i32(<5 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 4, i32 4, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <5 x i32> %arg0
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v5i32_fp4__v8i32_fp4(<5 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i64.v8i32(<5 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: operand 1 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v5i32(<8 x i32> %arg0, <5 x i32> %arg1, <4 x float> %arg2, i32 4, i32 4, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <5 x i32> %arg1
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_fp4__v5i32_fp4(<8 x i32> %arg0, <5 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v5i32(<8 x i32> %arg0, <5 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: operand 0 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v7i32.v8i32(<7 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 4, i32 4, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <7 x i32> %arg0
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v7i32_fp4__v8i32_fp4(<7 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i64.v8i32(<7 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: operand 1 must be 4, 6 or 8 element i32 vector
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v7i32(<8 x i32> %arg0, <7 x i32> %arg1, <4 x float> %arg2, i32 4, i32 4, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <7 x i32> %arg1
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_fp4__v7i32_fp4(<8 x i32> %arg0, <7 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v7i32(<8 x i32> %arg0, <7 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 4, ; cbsz
                                                                                      i32 4, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; --------------------------------------------------------------------
; Out of bounds format
; --------------------------------------------------------------------

; CHECK: invalid value for cbsz format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 9999, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: i32 9999
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_invalid0__v8i32_fp6(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 9999, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid value for blgp format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 0, i32 9999, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: i32 9999
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_fp8__v8i32_invalid0(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 9999, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid value for cbsz format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 5, i32 2, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: i32 5
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_invalid1__v8i32_fp6(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 5, ; cbsz
                                                                                      i32 2, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid value for blgp format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 0, i32 5, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: i32 5
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v8i32_fp8__v8i321_invalid(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 5, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid value for cbsz format
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 5, i32 5, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: i32 5
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v8i32_invalid__v8i32_invalid(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 5, ; cbsz
                                                                                      i32 5, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; --------------------------------------------------------------------
; Incorrect signature for format cases (IR vector too small)
; --------------------------------------------------------------------

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i32> %arg0
; CHECK-NEXT: i32 0
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v4i32_fp8__v8i32_fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i32> %arg1
; CHECK-NEXT: i32 0
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4__v8i32_fp8___v4i32_fp8(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <4 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i32> %arg0
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v4i32_fp8__v4i32_fp8(<4 x i32> %arg0, <4 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <4 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <6 x i32> %arg0
define <4 x float> @test_mfma_scale_f32_16x16x128_f8f6f4___v6i32_fp8__v6i32_fp8(<6 x i32> %arg0, <6 x i32> %arg1, <4 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <4 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <4 x float> %result
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <4 x i32> %arg0
; CHECK-NEXT: i32 0
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v4i32_fp8__v4i32_fp8(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> %arg0, <4 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}

; CHECK: invalid vector type for format
; CHECK-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 0, i32 0, i32 0, i32 %scale0, i32 0, i32 %scale1)
; CHECK-NEXT: <6 x i32> %arg0
; CHECK-NEXT: i32 0
define <16 x float> @test_mfma_scale_f32_32x32x64_f8f6f4___v6i32_fp8__v6i32_fp8(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2, i32 %scale0, i32 %scale1) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %arg0, <6 x i32> %arg1, <16 x float> %arg2,
                                                                                      i32 0, ; cbsz
                                                                                      i32 0, ; blgp
                                                                                      i32 0, i32 %scale0, i32 0, i32 %scale1)
  ret <16 x float> %result
}
