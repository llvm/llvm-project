; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -amdgpu-global-isel-new-legality -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
;
; Check that the intrinsic policy reaches a durable MachineIR opcode with
; the requested C/D register classes, ready for machine optimization passes.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)

define <4 x float> @automatic(float %a, float %b, <4 x float> %c) {
; ISEL-LABEL: name: automatic
; ISEL: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_vgprcd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  ret <4 x float> %d
}

define <4 x float> @require_agpr(float %a, float %b, <4 x float> %c) {
; ISEL-LABEL: name: require_agpr
; ISEL: :areg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_cd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ret <4 x float> %d
}

define <4 x float> @require_vgpr(float %a, float %b, <4 x float> %c) {
; ISEL-LABEL: name: require_vgpr
; ISEL: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_cd_vgprcd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) cd_regclass(vgpr)
  ret <4 x float> %d
}

define <4 x float> @zero_agpr(float %a, float %b) {
; ISEL-LABEL: name: zero_agpr
; ISEL: V_MFMA_F32_16X16X4F32_cd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ret <4 x float> %d
}

define <4 x float> @zero_vgpr(float %a, float %b) {
; ISEL-LABEL: name: zero_vgpr
; ISEL: V_MFMA_F32_16X16X4F32_cd_vgprcd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) cd_regclass(vgpr)
  ret <4 x float> %d
}

define <4 x float> @mixed_chain(float %a, float %b, <4 x float> %c) {
; ISEL-LABEL: name: mixed_chain
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64
; ISEL: V_MFMA_F32_16X16X4F32_cd_e64
; ISEL: V_MFMA_F32_16X16X4F32_cd_vgprcd_e64
  %d0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) cd_regclass(auto)
  %d1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %d0, i32 0, i32 0, i32 0) cd_regclass(agpr)
  %d2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %d1, i32 0, i32 0, i32 0) cd_regclass(vgpr)
  ret <4 x float> %d2
}

define <4 x float> @scaled_agpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; ISEL-LABEL: name: scaled_agpr
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_cd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 1, i32 %sa, i32 2, i32 %sb) cd_regclass(agpr)
  ret <4 x float> %d
}

define <4 x float> @scaled_vgpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; ISEL-LABEL: name: scaled_vgpr
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_cd_vgprcd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 1, i32 %sa, i32 2, i32 %sb) cd_regclass(vgpr)
  ret <4 x float> %d
}

define <4 x float> @unscaled_agpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
; ISEL-LABEL: name: unscaled_agpr
; ISEL: V_MFMA_F32_16X16X128_F8F6F4_f8_f8_cd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ret <4 x float> %d
}

define <4 x float> @unscaled_vgpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
; ISEL-LABEL: name: unscaled_vgpr
; ISEL: V_MFMA_F32_16X16X128_F8F6F4_f8_f8_cd_vgprcd_e64
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) cd_regclass(vgpr)
  ret <4 x float> %d
}

define <4 x float> @loop_agpr(float %a, float %b, <4 x float> %c, i32 %n, ptr addrspace(1) %out) {
; ISEL-LABEL: name: loop_agpr
; ISEL: V_MFMA_F32_16X16X4F32_cd_e64
  br label %loop
loop:
  %i = phi i32 [ 0, %0 ], [ %next, %loop ]
  %acc = phi <4 x float> [ %c, %0 ], [ %d, %loop ]
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %acc, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ; The result also has a VGPR-valued memory use.
  store volatile <4 x float> %d, ptr addrspace(1) %out
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret <4 x float> %d
}
