; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-0.ll  2>&1 | FileCheck %s -check-prefix=ERR-0
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-1.ll  2>&1 | FileCheck %s -check-prefix=ERR-1
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-2.ll  2>&1 | FileCheck %s -check-prefix=ERR-2
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-3.ll  2>&1 | FileCheck %s -check-prefix=ERR-3
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-4.ll  2>&1 | FileCheck %s -check-prefix=ERR-4
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-5.ll  2>&1 | FileCheck %s -check-prefix=ERR-5
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-6.ll  2>&1 | FileCheck %s -check-prefix=ERR-6
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-7.ll  2>&1 | FileCheck %s -check-prefix=ERR-7
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-8.ll  2>&1 | FileCheck %s -check-prefix=ERR-8
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-9.ll  2>&1 | FileCheck %s -check-prefix=ERR-9
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-10.ll 2>&1 | FileCheck %s -check-prefix=ERR-10
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-11.ll 2>&1 | FileCheck %s -check-prefix=ERR-11
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -filetype=null %t/err-12.ll 2>&1 | FileCheck %s -check-prefix=ERR-12

;--- err-0.ll
; ERR-0: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.bpermute.b32

define amdgpu_ps void @test_bpermute_b32(i32 %src0, i32 %src1, ptr addrspace(1) %out) {
bb:
  %res = call i32 @llvm.amdgcn.bpermute.b32(i32 %src0, i32 %src1)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

;--- err-1.ll
; ERR-1: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permute.pair.gensgpr.b32
define amdgpu_ps void @test_permute_pair_gensgpr_b32(i32 %src0, i64 %src1, ptr addrspace(1) %out) {
bb:
  %res = call i32 @llvm.amdgcn.permute.pair.gensgpr.b32(i32 %src0, i64 %src1)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

;--- err-2.ll
; ERR-2: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permute.pair.bcast.b32
define amdgpu_ps void @test_permute_pair_bcast_b32(i32 %src0, ptr addrspace(1) %out) {
bb:
  %res = call i32 @llvm.amdgcn.permute.pair.bcast.b32(i32 %src0, i32 2)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

;--- err-3.ll
; ERR-3: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permute.pair.2src.rotate.group.b32
define amdgpu_ps void @test_permute_pair_2src_rotate_group_b32(i32 %src0, i32 %src1, ptr addrspace(1) %out) {
bb:
  %res = call i32 @llvm.amdgcn.permute.pair.2src.rotate.group.b32(i32 %src0, i32 %src1, i32 2)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

;--- err-4.ll
; ERR-4: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permute.pair.2src.interleave.b64
define amdgpu_ps void @test_permute_pair_2src_interleave_b64(i32 %src0, i32 %src1, ptr addrspace(1) %out0, ptr addrspace(1) %out1) {
bb:
  %pair = call { i32, i32 } @llvm.amdgcn.permute.pair.2src.interleave.b64(i32 %src0, i32 %src1, i32 2)
  %dst0 = extractvalue { i32, i32 } %pair, 0
  %dst1 = extractvalue { i32, i32 } %pair, 1
  store i32 %dst0, ptr addrspace(1) %out0
  store i32 %dst1, ptr addrspace(1) %out1
  ret void
}

;--- err-5.ll
; ERR-5: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permute.pack.tensor.2src.b64
define amdgpu_ps void @test_permute_pack_tensor_2src_b64(i32 %src0, i32 %src1, ptr addrspace(1) %out0, ptr addrspace(1) %out1) {
bb:
  %pair = call { i32, i32 } @llvm.amdgcn.permute.pack.tensor.2src.b64(i32 %src0, i32 %src1, i32 2)
  %dst0 = extractvalue { i32, i32 } %pair, 0
  %dst1 = extractvalue { i32, i32 } %pair, 1
  store i32 %dst0, ptr addrspace(1) %out0
  store i32 %dst1, ptr addrspace(1) %out1
  ret void
}

;--- err-6.ll
; ERR-6: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.scale.bias.activate.f32
define amdgpu_ps void @test_scale_bias_activate_f32(float %ssrc, <4 x float> %acc_in, float %bias, ptr addrspace(1) %out) {
bb:
  %dst = call <4 x float> @llvm.amdgcn.scale.bias.activate.f32(<4 x float> %acc_in, float %ssrc, float %bias, i32 2, i1 1)
  store <4 x float> %dst, ptr addrspace(1) %out
  ret void
}

;--- err-7.ll
; ERR-7: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.uniform.scale.activate.f32
define amdgpu_ps void @test_uniform_scale_activate_f32(float %ssrc, <4 x float> %acc_in, ptr addrspace(1) %out) {
bb:
  %dst = call <4 x float> @llvm.amdgcn.uniform.scale.activate.f32(<4 x float> %acc_in, float %ssrc, i32 2, i1 1)
  store <4 x float> %dst, ptr addrspace(1) %out
  ret void
}

;--- err-8.ll
; ERR-8: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.convolve.f32.iu4.3x3
define amdgpu_ps void @test_convolve.f32_iu4_3x3_4x2(ptr addrspace(1) %out, <4 x float> %acc_in, <18 x i32> %weights, <3 x i32> %tensor_col_center, <3 x i32> %tensor_col_left, <3 x i32> %tensor_col_right) {
bb:
  %dst = call <4 x float> @llvm.amdgcn.convolve.f32.iu4.3x3.v4f32.v4f32.v18i32.v3i32(<4 x float> %acc_in, <18 x i32> %weights, <3 x i32> %tensor_col_center, <3 x i32> %tensor_col_left, <3 x i32> %tensor_col_right, i32 3, i1 1)
  store <4 x float> %dst, ptr addrspace(1) %out
  ret void
}

;--- err-9.ll
; ERR-9: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.clamp
define amdgpu_ps void @test_wmma_f32_16x16x16_fp8_fp8_clamp(<2 x i32> %A, <2 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.clamp(<2 x i32> %A, <2 x i32> %B, <8 x float> %C, i1 1)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

;--- err-10.ll
; ERR-10: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.swmma.f32.16x16x32.fp8.fp8.clamp
define amdgpu_ps void @test_swmma_f32_16x16x32_fp8_fp8_clamp(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i32 %Index, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.swmma.f32.16x16x32.fp8.fp8.clamp(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i32 %Index, i1 1, i1 1)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void
}

;--- err-11.ll
; ERR-11: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fma.from.tensor.f32.i4
define amdgpu_ps void @test_fma_from_tensor_f32_i4_dequant_disable_4x2(ptr addrspace(1) %out, <2 x float> %acc_in, i32 %resid_0, <2 x float> %scale) {
bb:
  %dst = call <2 x float> @llvm.amdgcn.fma.from.tensor.f32.i4.v2f32(<2 x float> %acc_in, i32 %resid_0, <2 x float> %scale, i32 2, i1 1)
  store <2 x float> %dst, ptr addrspace(1) %out
  ret void
}

;--- err-12.ll
; ERR-12: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.cvt.to.tensor.i4.f32
define amdgpu_ps void @test_cvt_to_tensor_i4_f32_4x2x16(<4 x float> %acc_in, i8 %scale, ptr addrspace(1) %out0) {
bb:
  %dest = call i32 @llvm.amdgcn.cvt.to.tensor.i4.f32.v4f32(<4 x float> %acc_in, i8 %scale, i32 3, i1 1)
  store i32 %dest, ptr addrspace(1) %out0
  ret void
}
