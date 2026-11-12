; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 < %s | llvm-mc -triple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu9.50-amd-amdhsa -mcpu=gfx950 -filetype=obj -o /dev/null

; The verifier accepts source vectors wider than the selected formats require.
; CodeGen must discard the unused high elements before instruction selection so
; that the assembly tuple widths agree with the formats. The llvm-mc RUN lines
; enforce the tuple widths because the assembler rejects a mismatch.

define <4 x float> @mfma_fp4_overwide(<8 x i32> %a, <8 x i32> %b,
                                      <4 x float> %c) {
; CHECK-LABEL: mfma_fp4_overwide:
; CHECK: v_mfma_f32_16x16x128_f8f6f4 {{.*}} cbsz:4 blgp:4
; IR-LABEL: define <4 x float> @mfma_fp4_overwide(
; IR: [[A:%.*]] = shufflevector <8 x i32> %a, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; IR-NEXT: [[B:%.*]] = shufflevector <8 x i32> %b, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; IR-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> [[A]], <4 x i32> [[B]], <4 x float> %c, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %result
}

define <16 x float> @mfma_32x32_fp4_overwide(<8 x i32> %a,
                                             <8 x i32> %b,
                                             <16 x float> %c) {
; CHECK-LABEL: mfma_32x32_fp4_overwide:
; CHECK: v_mfma_f32_32x32x64_f8f6f4 {{.*}} cbsz:4 blgp:4
; IR-LABEL: define <16 x float> @mfma_32x32_fp4_overwide(
; IR: [[A_32:%.*]] = shufflevector <8 x i32> %a, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; IR-NEXT: [[B_32:%.*]] = shufflevector <8 x i32> %b, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; IR-NEXT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> [[A_32]], <4 x i32> [[B_32]], <16 x float> %c, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <16 x float> %c, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret <16 x float> %result
}

define <4 x float> @mfma_bf6_bf8_overwide_a(<8 x i32> %a,
                                            <8 x i32> %b,
                                            <4 x float> %c) {
; CHECK-LABEL: mfma_bf6_bf8_overwide_a:
; CHECK: v_mfma_f32_16x16x128_f8f6f4 {{.*}} cbsz:3 blgp:1
; IR-LABEL: define <4 x float> @mfma_bf6_bf8_overwide_a(
; IR: [[MIXED_A:%.*]] = shufflevector <8 x i32> %a, <8 x i32> poison, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
; IR-NEXT: %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v8i32(<6 x i32> [[MIXED_A]], <8 x i32> %b, <4 x float> %c, i32 3, i32 1, i32 0, i32 0, i32 0, i32 0)
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 3, i32 1, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %result
}

declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <16 x float>, i32, i32, i32, i32, i32, i32)
