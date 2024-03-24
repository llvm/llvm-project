; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck %s

declare <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v4f32.v16f16(<16 x half>, <16 x half>, <4 x float>)

; Make sure we don't crash when trying to select modifiers in SelectVOP3PMods.

define amdgpu_cs void @xyz () {
; CHECK-LABEL: xyz:
; CHECK: v_wmma_f32_16x16x16_f16 v[0:3], v[0:7], v[0:7], v[0:3]

.entry:
  br label %loop
loop:
  %ld = load <8 x float>, ptr addrspace(5) null, align 32
  %in_shuffle = shufflevector <8 x float> %ld, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %wmma = call <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v4f32.v16f16(<16 x half> undef, <16 x half> undef, <4 x float> %in_shuffle)
  %out_shuffle = shufflevector <4 x float> %wmma, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  store <8 x float> %out_shuffle, ptr addrspace(5) null, align 32
  br i1 false, label %.exit, label %loop
.exit:
  ret void
}
