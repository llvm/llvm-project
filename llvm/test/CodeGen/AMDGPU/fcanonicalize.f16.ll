; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -mattr=-flat-for-global -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s

declare half @llvm.fabs.f16(half) #0
declare half @llvm.canonicalize.f16(half) #0
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #0
declare <2 x half> @llvm.canonicalize.v2f16(<2 x half>) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0


; GCN-LABEL: {{^}}v_test_canonicalize_var_f16:
; GCN: v_max_f16_e32 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @v_test_canonicalize_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_f16:
; GCN: v_max_f16_e64 [[REG:v[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_f16(half addrspace(1)* %out, i16 zeroext %val.arg) #1 {
  %val = bitcast i16 %val.arg to half
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_f16:
; GCN: v_max_f16_e64 [[REG:v[0-9]+]], |{{v[0-9]+}}|, |{{v[0-9]+}}|
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fabs_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_fabs_var_f16:
; GCN: v_max_f16_e64 [[REG:v[0-9]+]], -|{{v[0-9]+}}|, -|{{v[0-9]+}}|
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %val.fabs.fneg = fsub half -0.0, %val.fabs
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_var_f16:
; GCN: v_max_f16_e64 [[REG:v[0-9]+]], -{{v[0-9]+}}, -{{v[0-9]+}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_var_f16(half addrspace(1)* %out) #1 {
  %val = load half, half addrspace(1)* %out
  %val.fneg = fsub half -0.0, %val
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fneg)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff8000{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -0.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffbc00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -1.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4c00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_literal_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 16.0)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_default_denormals_fold_canonicalize_denormal0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_default_denormals_fold_canonicalize_denormal1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff83ff{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff83ff{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_f16(half addrspace(1)* %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7c00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C00)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -1 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -2 to half))
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7DFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFDFF)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_f16(half addrspace(1)* %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFC01)
  store half %canonicalized, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_var_v2f16:
; VI-DAG: v_max_f16_sdwa [[REG0:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_max_f16_e32 [[REG1:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; VI-NOT: v_and_b32

; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+$}}
; GFX9: buffer_store_dword [[REG]]
define amdgpu_kernel void @v_test_canonicalize_var_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; FIXME: Fold modifier
; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_v2f16:
; VI-DAG: v_bfe_u32
; VI-DAG: v_and_b32_e32 v{{[0-9]+}}, 0x7fff7fff, v{{[0-9]+}}
; VI: v_max_f16_sdwa [[REG0:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_max_f16_e32 [[REG1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI-NOT: 0xffff
; VI: v_or_b32

; GFX9: v_and_b32_e32 [[ABS:v[0-9]+]], 0x7fff7fff, v{{[0-9]+}}
; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], [[ABS]], [[ABS]]{{$}}
; GCN: buffer_store_dword
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
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, 0x80008000, v{{[0-9]+}}
; VI-DAG: v_max_f16_sdwa [[REG0:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_max_f16_e32 [[REG1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_or_b32

; GFX9: v_and_b32_e32 [[ABS:v[0-9]+]], 0x7fff7fff, v{{[0-9]+}}
; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], [[ABS]], [[ABS]] neg_lo:[1,1] neg_hi:[1,1]{{$}}
; GCN: buffer_store_dword
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
; VI:     v_xor_b32_e32 [[FNEG:v[0-9]+]], 0x80008000, v{{[0-9]+}}
; VI:     v_lshrrev_b32_e32 [[FNEGHI:v[0-9]+]], 16, [[FNEG]]
; VI-DAG: v_max_f16_sdwa [[REG1:v[0-9]+]], [[FNEG]], [[FNEGHI]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_max_f16_e32 [[REG0:v[0-9]+]], [[FNEG]], [[FNEG]]
; VI-NOT: 0xffff

; GFX9: v_pk_max_f16 [[REG:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}} neg_lo:[1,1] neg_hi:[1,1]{{$}}
; GFX9: buffer_store_dword [[REG]]
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
; GFX9: buffer_store_dword [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_v2f16(<2 x half> addrspace(1)* %out, i32 zeroext %val.arg) #1 {
  %val = bitcast i32 %val.arg to <2 x half>
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> zeroinitializer)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -0.0, half -0.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c003c00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 1.0, half 1.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbc00bc00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -1.0, half -1.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4c004c00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_literal_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 16.0, half 16.0>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff03ff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal0_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3ff03ff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_v2f16(<2 x half> addrspace(1)* %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal1_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x83ff83ff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x83ff83ff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_v2f16(<2 x half> addrspace(1)* %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7c007c00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C00, half 0xH7C00>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> bitcast (i32 -1 to <2 x half>))
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half bitcast (i16 -2 to half), half bitcast (i16 -2 to half)>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C01, half 0xH7C01>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7DFF, half 0xH7DFF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFDFF, half 0xHFDFF>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7e007e00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_v2f16(<2 x half> addrspace(1)* %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFC01, half 0xHFC01>)
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind "target-features"="-fp64-fp16-denormals" }
attributes #3 = { nounwind "target-features"="+fp64-fp16-denormals" }
