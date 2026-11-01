; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=0 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=0 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=IR
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=0 < %s | llvm-mc -triple=amdgpu12.50 -mcpu=gfx1250 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu12.50 -mcpu=gfx1250 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=0 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=STRICT-IR
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=1 -global-isel-abort=1 -print-after=amdgpu-lower-intrinsics -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=STRICT-IR
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=0 < %s | FileCheck %s --check-prefix=STRICT
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s --check-prefix=STRICT
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=0 < %s | llvm-mc -triple=amdgpu12.50s -mcpu=gfx1250 -filetype=obj -o /dev/null
; RUN: llc -O0 -mtriple=amdgpu12.50s -global-isel=1 -global-isel-abort=1 < %s | llvm-mc -triple=amdgpu12.50s -mcpu=gfx1250 -filetype=obj -o /dev/null

; The verifier accepts source vectors wider than the selected formats require.
; CodeGen must discard the unused high elements before instruction selection so
; that the assembly tuple widths agree with the formats. The llvm-mc RUN lines
; enforce the tuple widths because the assembler rejects a mismatch.

define <8 x float> @wmma_fp4_overwide(<16 x i32> %a, <16 x i32> %b,
                                      <8 x float> %c) {
; CHECK-LABEL: wmma_fp4_overwide:
; CHECK: v_wmma_f32_16x16x128_f8f6f4 {{.*}} matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
; IR-LABEL: define <8 x float> @wmma_fp4_overwide(
; IR: [[A:%.*]] = shufflevector <16 x i32> %a, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: [[B:%.*]] = shufflevector <16 x i32> %b, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v8i32.v8i32(i32 4, <8 x i32> [[A]], i32 4, <8 x i32> [[B]], i16 0, <8 x float> %c)
; STRICT-IR-LABEL: define <8 x float> @wmma_fp4_overwide(
; STRICT-IR: [[STRICT_A:%.*]] = shufflevector <16 x i32> %a, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; STRICT-IR-NEXT: [[STRICT_B:%.*]] = shufflevector <16 x i32> %b, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; STRICT-IR-NEXT: %result = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v8i32(i32 4, <8 x i32> [[STRICT_A]], i32 4, <8 x i32> [[STRICT_B]], i16 0, <8 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
; STRICT-LABEL: wmma_fp4_overwide:
; STRICT: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}} matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 4, <16 x i32> %a, i32 4, <16 x i32> %b, i16 0, <8 x float> %c)
  ret <8 x float> %result
}

define <8 x float> @wmma_scale_fp4_overwide(<16 x i32> %a,
                                            <16 x i32> %b,
                                            <8 x float> %c) {
; CHECK-LABEL: wmma_scale_fp4_overwide:
; CHECK: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}} matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
; IR-LABEL: define <8 x float> @wmma_scale_fp4_overwide(
; IR: [[SCALE_A:%.*]] = shufflevector <16 x i32> %a, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: [[SCALE_B:%.*]] = shufflevector <16 x i32> %b, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: %result = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v8i32(i32 4, <8 x i32> [[SCALE_A]], i32 4, <8 x i32> [[SCALE_B]], i16 0, <8 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  %result = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 4, <16 x i32> %a, i32 4, <16 x i32> %b, i16 0, <8 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i1 false, i1 false)
  ret <8 x float> %result
}

define <8 x float> @wmma_scale16_fp4_overwide(<16 x i32> %a,
                                              <16 x i32> %b,
                                              <8 x float> %c) {
; CHECK-LABEL: wmma_scale16_fp4_overwide:
; CHECK: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}} matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
; IR-LABEL: define <8 x float> @wmma_scale16_fp4_overwide(
; IR: [[SCALE16_A:%.*]] = shufflevector <16 x i32> %a, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: [[SCALE16_B:%.*]] = shufflevector <16 x i32> %b, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; IR-NEXT: %result = call <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v8i32.v8i32(i32 4, <8 x i32> [[SCALE16_A]], i32 4, <8 x i32> [[SCALE16_B]], i16 0, <8 x float> %c, i32 0, i32 0, i64 0, i32 0, i32 0, i64 0, i1 false, i1 false)
  %result = call <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 4, <16 x i32> %a, i32 4, <16 x i32> %b, i16 0, <8 x float> %c, i32 0, i32 0, i64 0, i32 0, i32 0, i64 0, i1 false, i1 false)
  ret <8 x float> %result
}

define <8 x float> @wmma_bf8_fp6_overwide_b(<16 x i32> %a,
                                            <16 x i32> %b,
                                            <8 x float> %c) {
; CHECK-LABEL: wmma_bf8_fp6_overwide_b:
; CHECK: v_wmma_f32_16x16x128_f8f6f4 {{.*}} matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6
; IR-LABEL: define <8 x float> @wmma_bf8_fp6_overwide_b(
; IR: [[MIXED_B:%.*]] = shufflevector <16 x i32> %b, <16 x i32> poison, <12 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
; IR-NEXT: %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v12i32(i32 1, <16 x i32> %a, i32 2, <12 x i32> [[MIXED_B]], i16 0, <8 x float> %c)
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 1, <16 x i32> %a, i32 2, <16 x i32> %b, i16 0, <8 x float> %c)
  ret <8 x float> %result
}

declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>, i32, i32, i32, i32, i32, i32, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>, i32, i32, i64, i32, i32, i64, i1, i1)
