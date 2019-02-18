; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s

declare half @llvm.fabs.f16(half) #0
declare half @llvm.canonicalize.f16(half) #0
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #0
declare <2 x half> @llvm.canonicalize.v2f16(<2 x half>) #0
declare <3 x half> @llvm.canonicalize.v3f16(<3 x half>) #0
declare <4 x half> @llvm.canonicalize.v4f16(<4 x half>) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0

; GCN-LABEL: {{^}}test_fold_canonicalize_undef_value_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_undef_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half undef)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_var_f16:
; GFX89: v_max_f16_e32 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]

; CI: v_cvt_f32_f16_e32
; CI: v_mul_f32_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @v_test_canonicalize_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_f16:
; GFX89: v_max_f16_e64 [[REG:v[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_f16(half addrspace(1)* %out, i16 zeroext %val.arg) #1 {
  %val = bitcast i16 %val.arg to half
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_build_vector_v2f16:
; GFX9: v_and_b32_e32 v0, 0xffff, v0
; GFX9-NEXT: v_lshl_or_b32 v0, v1, 16, v0
; GFX9-NEXT: v_pk_max_f16 v0, v0, v0

; VI: v_max_f16_sdwa v1, v1, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_max_f16_e32 v0, v0, v0
; VI: v_or_b32_e32 v0, v0, v1
define <2 x half> @v_test_canonicalize_build_vector_v2f16(half %lo, half %hi) #1 {
  %ins0 = insertelement <2 x half> undef, half %lo, i32 0
  %ins1 = insertelement <2 x half> %ins0, half %hi, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins1)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_f16:
; GFX89: v_max_f16_e64 [[REG:v[0-9]+]], |{{v[0-9]+}}|, |{{v[0-9]+}}|
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fabs_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_fabs_var_f16:
; GFX89: v_max_f16_e64 [[REG:v[0-9]+]], -|{{v[0-9]+}}|, -|{{v[0-9]+}}|
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]

; CI: v_cvt_f32_f16_e64 v{{[0-9]+}}, -|v{{[0-9]+}}|
; CI: v_mul_f32_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %val.fabs.fneg = fsub half -0.0, %val.fabs
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_var_f16:
; GFX89: v_max_f16_e64 [[REG:v[0-9]+]], -{{v[0-9]+}}, -{{v[0-9]+}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]

; CI: v_cvt_f32_f16_e64 {{v[0-9]+}}, -{{v[0-9]+}}
; CI: v_mul_f32_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @v_test_canonicalize_fneg_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fneg = fsub half -0.0, %val
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_no_denormals_canonicalize_fneg_var_f16:
; GFX89: v_mul_f16_e32 [[REG:v[0-9]+]], -1.0, v{{[0-9]+}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @v_test_no_denormals_canonicalize_fneg_var_f16(half addrspace(1)* %out) #2 {
  %val = load half, half addrspace(1)* %out
  %val.fneg = fsub half -0.0, %val
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_no_denormals_canonicalize_fneg_fabs_var_f16:
; GFX89: v_mul_f16_e64 [[REG:v[0-9]+]], -1.0, |v{{[0-9]+}}|
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]

; CI: v_cvt_f32_f16_e64 {{v[0-9]+}}, -|{{v[0-9]+}}|
; CI: v_mul_f32_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @v_test_no_denormals_canonicalize_fneg_fabs_var_f16(half addrspace(1)* %out) #2 {
  %val = load half, half addrspace(1)* %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %val.fabs.fneg = fsub half -0.0, %val.fabs
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff8000{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffbc00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4c00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_literal_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 16.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_default_denormals_fold_canonicalize_denormal0_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_default_denormals_fold_canonicalize_denormal1_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff83ff{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff83ff{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7c00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C00)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -1 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -2 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7DFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFDFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFC01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_var_v2f16:
; VI-DAG: v_max_f16_sdwa [[REG0:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[REG1:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; VI-NOT: v_and_b32

; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+$}}
; GFX9: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @v_test_canonicalize_var_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_v2f16:
; VI: v_max_f16_sdwa [[REG0:v[0-9]+]], |v{{[0-9]+}}|, |v{{[0-9]+}}| dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_max_f16_e64 [[REG1:v[0-9]+]], |v{{[0-9]+}}|, |v{{[0-9]+}}|
; VI-NOT: 0xffff
; VI: v_or_b32

; GFX9: v_and_b32_e32 [[ABS:v[0-9]+]], 0x7fff7fff, v{{[0-9]+}}
; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], [[ABS]], [[ABS]]{{$}}
; GFX89: {{flat|global}}_store_dword
define amdgpu_kernel void @v_test_canonicalize_fabs_var_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %val.fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val.fabs)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_fabs_var_v2f16:
; VI-DAG: v_max_f16_sdwa [[REG0:v[0-9]+]], -|v{{[0-9]+}}|, -|v{{[0-9]+}}| dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e64 [[REG1:v[0-9]+]], -|v{{[0-9]+}}|, -|v{{[0-9]+}}|
; VI: v_or_b32

; GFX9: v_and_b32_e32 [[ABS:v[0-9]+]], 0x7fff7fff, v{{[0-9]+}}
; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], [[ABS]], [[ABS]] neg_lo:[1,1] neg_hi:[1,1]{{$}}
; GFX89: {{flat|global}}_store_dword

; CI: v_cvt_f32_f16
; CI: v_cvt_f32_f16
; CI: v_mul_f32_e32 v{{[0-9]+}}, 1.0
; CI: v_mul_f32_e32 v{{[0-9]+}}, 1.0
define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %val.fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %val.fabs.fneg = fsub <2 x half> <half -0.0, half -0.0>, %val.fabs
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val.fabs.fneg)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_var_v2f16:
; VI-DAG: v_max_f16_sdwa [[REG1:v[0-9]+]], -v{{[0-9]+}}, -v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e64 [[REG0:v[0-9]+]], -v{{[0-9]+}}, -v{{[0-9]+}}
; VI-NOT: 0xffff

; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}} neg_lo:[1,1] neg_hi:[1,1]{{$}}
; GFX9: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_var_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %fneg.val = fsub <2 x half> <half -0.0, half -0.0>, %val
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %fneg.val)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_v2f16:
; VI: v_max_f16_sdwa [[REG0:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_max_f16_e64 [[REG1:v[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; VI-NOT: v_and_b32

; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], {{s[0-9]+}}, {{s[0-9]+$}}
; GFX9: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_v2f16(<2 x half> addrspace(1)* %out, i32 zeroext %val.arg) #1 {
  %val = bitcast i32 %val.arg to <2 x half>
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> zeroinitializer)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -0.0, half -0.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c003c00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 1.0, half 1.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbc00bc00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -1.0, half -1.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4c004c00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_literal_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 16.0, half 16.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal0_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff03ff{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff03ff{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_v2f16(<2 x half> addrspace(1)* %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal1_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x83ff83ff{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x83ff83ff{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_v2f16(<2 x half> addrspace(1)* %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7c007c00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C00, half 0xH7C00>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> bitcast (i32 -1 to <2 x half>))
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half bitcast (i16 -2 to half), half bitcast (i16 -2 to half)>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C01, half 0xH7C01>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7DFF, half 0xH7DFF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFDFF, half 0xHFDFF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_v2f16:
; GFX89: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFC01, half 0xHFC01>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; FIXME: Extra 4th component handled
; GCN-LABEL: {{^}}v_test_canonicalize_var_v3f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 v0, v0, v0
; GFX9-NEXT: v_pk_max_f16 v1, v1, v1
; GFX9-NEXT: s_setpc_b64

; VI-DAG: v_max_f16_sdwa [[CANON_ELT1:v[0-9]+]], v0, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON_ELT0:v[0-9]+]], v0, v0
; VI-DAG: v_max_f16_e32 v1, v1, v1
; VI-DAG: v_or_b32_e32 v0, [[CANON_ELT0]], [[CANON_ELT1]]

; VI: s_setpc_b64
define <3 x half> @v_test_canonicalize_var_v3f16(<3 x half> %val) #1 {
  %canonicalized = call <3 x half> @llvm.canonicalize.v3f16(<3 x half> %val)
  ret <3 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_var_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 v0, v0, v0
; GFX9-NEXT: v_pk_max_f16 v1, v1, v1
; GFX9-NEXT: s_setpc_b64

; VI-DAG: v_max_f16_sdwa [[CANON_ELT3:v[0-9]+]], v1, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON_ELT2:v[0-9]+]], v1, v1
; VI-DAG: v_max_f16_sdwa [[CANON_ELT1:v[0-9]+]], v0, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON_ELT0:v[0-9]+]], v0, v0
; VI-DAG: v_or_b32_e32 v0, [[CANON_ELT0]], [[CANON_ELT1]]
; VI-DAG: v_or_b32_e32 v1, [[CANON_ELT2]], [[CANON_ELT3]]
; VI: s_setpc_b64
define <4 x half> @v_test_canonicalize_var_v4f16(<4 x half> %val) #1 {
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %val)
  ret <4 x half> %canonicalized
}

; GCN-LABEL: {{^}}s_test_canonicalize_undef_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00
; GFX89: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[REG]]
define amdgpu_kernel void @s_test_canonicalize_undef_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> undef)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_reg_undef_v2f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_max_f16_e32 v0, v0, v0
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v0
; GFX9-NEXT: s_setpc_b64

; High bits known zero
; FIXME: Should also be true on gfx9 by default?
; VI: s_waitcnt
; VI-NEXT: v_max_f16_e32 v0, v0, v0
; VI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_reg_undef_v2f16(half %val) #1 {
  %vec = insertelement <2 x half> undef, half %val, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_undef_reg_v2f16:
; GFX89: s_waitcnt
; GFX89-NEXT: v_max_f16_sdwa v0, v0, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; GFX89-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_undef_reg_v2f16(half %val) #1 {
  %vec = insertelement <2 x half> undef, half %val, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_undef_lo_imm_hi_v2f16:
; GCN: s_waitcnt
; GFX89-NEXT: v_mov_b32_e32 v0, 0x3c003c00
; GFX89-NEXT: s_setpc_b64

; CI-NEXT: v_mov_b32_e32 v0, 0x7fc00000
; CI-NEXT: v_mov_b32_e32 v1, 1.0
; CI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_undef_lo_imm_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 1.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_imm_lo_undef_hi_v2f16:
; GCN: s_waitcnt
; GFX89-NEXT: v_mov_b32_e32 v0, 0x3c003c00
; GFX89-NEXT: s_setpc_b64

; CI-NEXT: v_mov_b32_e32 v0, 1.0
; CI-NEXT: v_mov_b32_e32 v1, 0x7fc00000
; CI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_imm_lo_undef_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 1.0, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_undef_lo_k_hi_v2f16:
; GCN: s_waitcnt
; GFX89-NEXT: v_mov_b32_e32 v0, 0x4c004c00
; GFX89-NEXT: s_setpc_b64

; CI-NEXT: v_mov_b32_e32 v0, 0x7fc00000
; CI-NEXT: v_mov_b32_e32 v1, 0x41800000
; CI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_undef_lo_k_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 16.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_k_lo_undef_hi_v2f16:
; GCN: s_waitcnt
; GFX89-NEXT: v_mov_b32_e32 v0, 0x4c004c00
; GFX89-NEXT: s_setpc_b64

; CI-NEXT: v_mov_b32_e32 v0, 0x41800000
; CI-NEXT: v_mov_b32_e32 v1, 0x7fc00000
; CI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_k_lo_undef_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 16.0, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_reg_k_v2f16:
; GFX9: s_waitcnt
; GFX9-DAG: v_max_f16_e32 v0, v0, v0
; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x4000
; GFX9: v_and_b32_e32 v0, 0xffff, v0
; GFX9: v_lshl_or_b32 v0, [[K]], 16, v0
; GFX9: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_max_f16_e32 v0, v0, v0
; VI-NEXT: v_or_b32_e32 v0, 2.0, v0
; VI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_reg_k_v2f16(half %val) #1 {
  %vec0 = insertelement <2 x half> undef, half %val, i32 0
  %vec1 = insertelement <2 x half> %vec0, half 2.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec1)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_k_reg_v2f16:
; GFX9: v_max_f16_e32 v0, v0, v0
; GFX9: v_mov_b32_e32 [[K:v[0-9]+]], 0x4000
; GFX9: v_lshl_or_b32 v0, v0, 16, [[K]]
; GFX9: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_max_f16_sdwa v0, v0, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-NEXT: v_or_b32_e32 v0, 0x4000, v0
; VI-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_k_reg_v2f16(half %val) #1 {
  %vec0 = insertelement <2 x half> undef, half 2.0, i32 0
  %vec1 = insertelement <2 x half> %vec0, half %val, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec1)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}s_test_canonicalize_undef_v4f16:
; GCN: v_mov_b32_e32 v0, 0x7e007e00
; GCN: v_mov_b32_e32 v1, v0
define amdgpu_kernel void @s_test_canonicalize_undef_v4f16(<4 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> undef)
  store <4 x half> %canonicalized, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_reg_undef_undef_undef_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_max_f16_e32 v0, v0, v0
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v0
; GFX9-NEXT: v_mov_b32_e32 v1, 0
; GFX9-NEXT: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_max_f16_e32 v0, v0, v0
; VI-NEXT: v_or_b32_e32 v0, 0x7e000000, v0
; VI-NEXT: v_mov_b32_e32 v1, 0x7e007e00
; VI-NEXT: s_setpc_b64
define <4 x half> @v_test_canonicalize_reg_undef_undef_undef_v4f16(half %val) #1 {
  %vec = insertelement <4 x half> undef, half %val, i32 0
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec)
  ret <4 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_reg_reg_undef_undef_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v0
; GFX9-NEXT: v_lshl_or_b32 v0, v1, 16, v0
; GFX9-NEXT: v_pk_max_f16 v0, v0, v0
; GFX9-NEXT: v_mov_b32_e32 v1, 0
; GFX9-NEXT: s_setpc_b64

; VI: s_waitcnt
; VI-DAG: v_max_f16_e32 v0, v0, v0
; VI-DAG: v_max_f16_sdwa v1, v1, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32_e32 v0, v0, v1
; VI-NEXT: v_mov_b32_e32 v1, 0x7e007e00
; VI-NEXT: s_setpc_b64
define <4 x half> @v_test_canonicalize_reg_reg_undef_undef_v4f16(half %val0, half %val1) #1 {
  %vec0 = insertelement <4 x half> undef, half %val0, i32 0
  %vec1 = insertelement <4 x half> %vec0, half %val1, i32 1
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec1)
  ret <4 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_reg_undef_reg_reg_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff
; GFX9-NEXT: v_and_b32_e32 v1, [[MASK]], v1
; GFX9-NEXT: v_max_f16_e32 v0, v0, v0
; GFX9-NEXT: v_lshl_or_b32 v1, v2, 16, v1
; GFX9-NEXT: v_and_b32_e32 v0, [[MASK]], v0
; GFX9-NEXT: v_pk_max_f16 v1, v1, v1
; GFX9-NEXT: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_max_f16_e32 v0, v0, v0
; VI-NEXT: v_max_f16_e32 v1, v1, v1
; VI-NEXT: v_max_f16_sdwa v2, v2, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-NEXT: v_or_b32_e32 v0, 0x7e000000, v0
; VI-NEXT: v_or_b32_e32 v1, v1, v2
; VI-NEXT: s_setpc_b64
define <4 x half> @v_test_canonicalize_reg_undef_reg_reg_v4f16(half %val0, half %val1, half %val2) #1 {
  %vec0 = insertelement <4 x half> undef, half %val0, i32 0
  %vec1 = insertelement <4 x half> %vec0, half %val1, i32 2
  %vec2 = insertelement <4 x half> %vec1, half %val2, i32 3
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec2)
  ret <4 x half> %canonicalized
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind "target-features"="-fp64-fp16-denormals" }
attributes #3 = { nounwind "target-features"="+fp64-fp16-denormals" }
