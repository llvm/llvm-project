; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -amdgpu-disable-rewrite-mfma-form-sched-stage=true -verify-machineinstrs -stop-after=amdgpu-prepare-agpr-alloc < %s | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -amdgpu-mfma-vgpr-form=0 -verify-machineinstrs -stop-after=amdgpu-prepare-agpr-alloc < %s | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -O0 -verify-machineinstrs -stop-after=amdgpu-prepare-agpr-alloc < %s | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -O0 -verify-machineinstrs -stop-after=amdgpu-prepare-agpr-alloc < %s | FileCheck %s --check-prefix=HARD
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -amdgpu-global-isel-new-legality -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -amdgpu-mfma-vgpr-form=0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=FORCE-AGPR
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -amdgpu-disable-rewrite-mfma-form-sched-stage=false -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel < %s | llc -x mir -mtriple=amdgpu9.50-amd-amdhsa -run-pass=machine-cse -verify-machineinstrs -o - | FileCheck %s --check-prefix=CSE
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -verify-machineinstrs -stop-after=finalize-isel < %s | llc -x mir -mtriple=amdgpu9.50-amd-amdhsa -run-pass=machine-cse -verify-machineinstrs -o - | FileCheck %s --check-prefix=CSE
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel < %s | llc -x mir -mtriple=amdgpu9.50-amd-amdhsa -run-pass=none -verify-machineinstrs -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 -global-isel-abort=1 -amdgpu-global-isel-new-legality -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
;
; Check that policies preserve the original instruction selection and survive
; as distinct target-index operands for the later AGPR allocation passes.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)

define <4 x float> @automatic(float %a, float %b, <4 x float> %c) {
; ASM-LABEL: automatic:
; ASM: v_mfma_f32_16x16x4_f32 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; FORCE-AGPR-LABEL: automatic:
; FORCE-AGPR: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}]
; An unconstrained instruction has no policy operand, even when the selected
; form uses VGPR C/D. This must survive MIR round trips and MachineIR CSE.
; ISEL-LABEL: name: automatic
; ISEL-NOT: target-index(amdgpu-mfma-cd-
; ISEL: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, implicit $mode, implicit $exec{{$}}
; ISEL-NOT: target-index(amdgpu-mfma-cd-
; CSE-LABEL: name: automatic
; CSE-NOT: target-index(amdgpu-mfma-cd-
; CSE: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, implicit $mode, implicit $exec{{$}}
; CSE-NOT: target-index(amdgpu-mfma-cd-
; CSE: ...
; HARD-LABEL: name: automatic
; HARD-NOT: target-index(amdgpu-mfma-cd-
; HARD: V_MFMA_F32_16X16X4F32{{[^ ]*}}
; HARD-NOT: target-index(amdgpu-mfma-cd-
; HARD: ...
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  ret <4 x float> %d
}

define <4 x float> @require_agpr(float %a, float %b, <4 x float> %c) {
; ASM-LABEL: require_agpr:
; ASM: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; FORCE-AGPR-LABEL: require_agpr:
; FORCE-AGPR: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}]
; HARD-LABEL: name: require_agpr
; HARD: V_MFMA_F32_16X16X4F32_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
; ISEL-LABEL: name: require_agpr
; ISEL: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @require_vgpr(float %a, float %b, <4 x float> %c) {
; ASM-LABEL: require_vgpr:
; ASM: v_mfma_f32_16x16x4_f32 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; FORCE-AGPR-LABEL: require_vgpr:
; FORCE-AGPR: v_mfma_f32_16x16x4_f32 v[{{[0-9]+:[0-9]+}}], {{.*}}, v[{{[0-9]+:[0-9]+}}]
; HARD-LABEL: name: require_vgpr
; HARD: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
; ISEL-LABEL: name: require_vgpr
; ISEL: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

define <4 x float> @zero_agpr(float %a, float %b) {
; ASM-LABEL: zero_agpr:
; ASM: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0
; ISEL-LABEL: name: zero_agpr
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @zero_vgpr(float %a, float %b) {
; ASM-LABEL: zero_vgpr:
; ASM: v_mfma_f32_16x16x4_f32 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, 0
; ISEL-LABEL: name: zero_vgpr
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

define <4 x float> @mixed_chain(float %a, float %b, <4 x float> %c) {
; ASM-LABEL: mixed_chain:
; ASM: v_mfma_f32_16x16x4_f32 {{[av]}}[{{[0-9]+:[0-9]+}}], {{.*}}
; ASM: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}]
; ASM: v_accvgpr_read_b32
; ASM: v_mfma_f32_16x16x4_f32 v[{{[0-9]+:[0-9]+}}], {{.*}}, v[{{[0-9]+:[0-9]+}}]
; ISEL-LABEL: name: mixed_chain
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  %d1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %d0, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  %d2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %d1, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d2
}

define <4 x float> @scaled_agpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; ASM-LABEL: scaled_agpr:
; ASM: v_mfma_scale_f32_16x16x128_f8f6f4 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}} op_sel:[1,0,0] op_sel_hi:[0,1,0]
; HARD-LABEL: name: scaled_agpr
; HARD: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
; ISEL-LABEL: name: scaled_agpr
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 1, i32 %sa, i32 2, i32 %sb) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @scaled_vgpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; ASM-LABEL: scaled_vgpr:
; ASM: v_mfma_scale_f32_16x16x128_f8f6f4 v[{{[0-9]+:[0-9]+}}], {{.*}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}} op_sel:[1,0,0] op_sel_hi:[0,1,0]
; HARD-LABEL: name: scaled_vgpr
; HARD: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
; ISEL-LABEL: name: scaled_vgpr
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 1, i32 %sa, i32 2, i32 %sb) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

define <4 x float> @unscaled_agpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
; ASM-LABEL: unscaled_agpr:
; ASM: v_mfma_f32_16x16x128_f8f6f4 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}]
; ISEL-LABEL: name: unscaled_agpr
; ISEL: V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @unscaled_vgpr(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
; ASM-LABEL: unscaled_vgpr:
; ASM: v_mfma_f32_16x16x128_f8f6f4 v[{{[0-9]+:[0-9]+}}], {{.*}}, v[{{[0-9]+:[0-9]+}}]
; ISEL-LABEL: name: unscaled_vgpr
; ISEL: V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

define <4 x float> @loop_agpr(float %a, float %b, <4 x float> %c, i32 %n, ptr addrspace(1) %out) {
; ASM-LABEL: loop_agpr:
; ASM: v_mfma_f32_16x16x4_f32 a[{{[0-9]+:[0-9]+}}], {{.*}}, a[{{[0-9]+:[0-9]+}}]
; ISEL-LABEL: name: loop_agpr
; ISEL: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  br label %loop
loop:
  %i = phi i32 [ 0, %0 ], [ %next, %loop ]
  %acc = phi <4 x float> [ %c, %0 ], [ %d, %loop ]
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %acc, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ; The result also has a VGPR-valued memory use.
  store volatile <4 x float> %d, ptr addrspace(1) %out
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret <4 x float> %d
}

; Exercise tied forms and the existing source-width selection patterns.
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v4i32(<6 x i32>, <4 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)

define <16 x float> @tied_agpr(float %a, float %b, <16 x float> %c) {
; HARD-LABEL: name: tied_agpr
; HARD: V_MFMA_F32_32X32X2F32_mac_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
; ISEL-LABEL: name: tied_agpr
; ISEL: :vreg_512{{[^ ]*}} = V_MFMA_F32_32X32X2F32_mac_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %a, float %b, <16 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <16 x float> %d
}

define <16 x float> @tied_vgpr(float %a, float %b, <16 x float> %c) {
; HARD-LABEL: name: tied_vgpr
; HARD: V_MFMA_F32_32X32X2F32_mac_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
; ISEL-LABEL: name: tied_vgpr
; ISEL: :vreg_512{{[^ ]*}} = V_MFMA_F32_32X32X2F32_mac_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %a, float %b, <16 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <16 x float> %d
}

define <4 x float> @scaled_f6_f4_agpr(<6 x i32> %a, <4 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; ISEL-LABEL: name: scaled_f6_f4_agpr
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64 {{.*}}, 2, 4, {{.*}}, 4, 8, target-index(amdgpu-mfma-cd-agpr), implicit
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v4i32(<6 x i32> %a, <4 x i32> %b, <4 x float> %c, i32 2, i32 4, i32 1, i32 %sa, i32 2, i32 %sb) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @unscaled_f6_f4_vgpr(<6 x i32> %a, <4 x i32> %b, <4 x float> %c) {
; ISEL-LABEL: name: unscaled_f6_f4_vgpr
; ISEL: V_MFMA_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64 {{.*}}, 2, 4, target-index(amdgpu-mfma-cd-vgpr), implicit
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v4i32(<6 x i32> %a, <4 x i32> %b, <4 x float> %c, i32 2, i32 4, i32 1, i32 0, i32 2, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

; The same selected opcode must retain the distinction between automatic and
; required VGPR C/D, both during DAG selection and after MachineIR CSE.
define <4 x float> @same_form_mixed_policy(float %a, float %b, <4 x float> %c) {
; ISEL-LABEL: name: same_form_mixed_policy
; ISEL-DAG: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, implicit
; ISEL-DAG: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, target-index(amdgpu-mfma-cd-vgpr), implicit
; CSE-LABEL: name: same_form_mixed_policy
; CSE-DAG: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, implicit
; CSE-DAG: V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, 0, 0, target-index(amdgpu-mfma-cd-vgpr), implicit
  %auto = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  %hard = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  %sum = fadd <4 x float> %auto, %hard
  ret <4 x float> %sum
}


; Scaled MFMA also omits the policy operand when the attribute is absent.
define <4 x float> @scaled_automatic(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %scale_a, i32 %scale_b) {
; ISEL-LABEL: name: scaled_automatic
; ISEL-NOT: target-index(amdgpu-mfma-cd-
; ISEL: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64
; ISEL-NOT: target-index(amdgpu-mfma-cd-
; ISEL: ...
; CSE-LABEL: name: scaled_automatic
; CSE-NOT: target-index(amdgpu-mfma-cd-
; CSE: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64
; CSE-NOT: target-index(amdgpu-mfma-cd-
; CSE: ...
; HARD-LABEL: name: scaled_automatic
; HARD-NOT: target-index(amdgpu-mfma-cd-
; HARD: V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8{{[^ ]*}}
; HARD-NOT: target-index(amdgpu-mfma-cd-
; HARD: ...
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 1, i32 %scale_a, i32 2, i32 %scale_b)
  ret <4 x float> %d
}

; optnone must not skip hard constraints.
define <4 x float> @optnone_agpr(float %a, float %b, <4 x float> %c) noinline optnone {
; HARD-LABEL: name: optnone_agpr
; HARD: :areg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-agpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

define <4 x float> @optnone_vgpr(float %a, float %b, <4 x float> %c) noinline optnone {
; HARD-LABEL: name: optnone_vgpr
; HARD: :vreg_128{{[^ ]*}} = V_MFMA_F32_16X16X4F32_vgprcd_e64 {{.*}}, 0, {{.*}}target-index(amdgpu-mfma-cd-vgpr)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}
