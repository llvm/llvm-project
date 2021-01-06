; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,SI,FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,VI,GFX8_9_10,FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9_10,GFX8_9_10,FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX10,GFX9_10,GFX8_9_10,FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cypress -verify-machineinstrs < %s | FileCheck --check-prefixes=EG,FUNC %s

; FUNC-LABEL: {{^}}v_test_imin_sle_i32:
; GCN: v_min_i32_e32

; EG: MIN_INT
define amdgpu_kernel void @v_test_imin_sle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i32:
; GCN: s_min_i32

; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v1i32:
; GCN: s_min_i32

; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) #0 {
  %cmp = icmp sle <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i32:
; GCN: s_min_i32
; GCN: s_min_i32
; GCN: s_min_i32
; GCN: s_min_i32

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) #0 {
  %cmp = icmp sle <4 x i32> %a, %b
  %val = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %val, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i8:
; GCN: s_load_dword
; GCN: s_load_dword
; GCN: s_sext_i32_i8
; GCN: s_sext_i32_i8
; GCN: s_min_i32
define amdgpu_kernel void @s_test_imin_sle_i8(i8 addrspace(1)* %out, [8 x i32], i8 %a, [8 x i32], i8 %b) #0 {
  %cmp = icmp sle i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out
  ret void
}

; FIXME: Why vector and sdwa for last element?
; FUNC-LABEL: {{^}}s_test_imin_sle_v4i8:
; GCN-DAG: s_load_dwordx2
; GCN-DAG: s_load_dword s
; GCN-DAG: s_load_dword s
; GCN-NOT: _load_

; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32

; VI: s_min_i32
; VI: s_min_i32
; VI: s_min_i32
; VI: v_min_i32_sdwa

; GFX9_10: v_min_i16
; GFX9_10: v_min_i16
; GFX9_10: v_min_i16
; GFX9_10: v_min_i16

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_v4i8(<4 x i8> addrspace(1)* %out, [8 x i32], <4 x i8> %a, [8 x i32], <4 x i8> %b) #0 {
  %cmp = icmp sle <4 x i8> %a, %b
  %val = select <4 x i1> %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v2i16:
; GCN: s_load_dword s
; GCN: s_load_dword s

; SI: s_ashr_i32
; SI: s_ashr_i32
; SI: s_sext_i32_i16
; SI: s_sext_i32_i16
; SI: s_min_i32
; SI: s_min_i32

; VI: s_sext_i32_i16
; VI: s_sext_i32_i16
; VI: s_min_i32
; VI: s_min_i32

; GFX9_10: v_pk_min_i16

; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp sle <2 x i16> %a, %b
  %val = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %val, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i16:
; SI-NOT: buffer_load
; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32

; VI: s_min_i32
; VI: s_min_i32
; VI: s_min_i32
; VI: s_min_i32

; GFX9_10: v_pk_min_i16
; GFX9_10: v_pk_min_i16

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> %a, <4 x i16> %b) #0 {
  %cmp = icmp sle <4 x i16> %a, %b
  %val = select <4 x i1> %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %val, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @v_test_imin_slt_i32
; GCN: v_min_i32_e32

; EG: MIN_INT
define amdgpu_kernel void @v_test_imin_slt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %bptr, i32 %tid
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; FUNC-LABEL: @v_test_imin_slt_i16
; SI: v_min_i32_e32

; GFX8_9: v_min_i16_e32
; GFX10:  v_min_i16_e64

; EG: MIN_INT
define amdgpu_kernel void @v_test_imin_slt_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i16, i16 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr inbounds i16, i16 addrspace(1)* %bptr, i32 %tid
  %out.gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid

  %a = load i16, i16 addrspace(1)* %a.gep
  %b = load i16, i16 addrspace(1)* %b.gep
  %cmp = icmp slt i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: @s_test_imin_slt_i32
; GCN: s_min_i32

; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_slt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_slt_v2i32:
; GCN: s_min_i32
; GCN: s_min_i32

; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_slt_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %cmp = icmp slt <2 x i32> %a, %b
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_slt_imm_i32:
; GCN: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8

; EG: MIN_INT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_imin_slt_imm_i32(i32 addrspace(1)* %out, i32 %a) #0 {
  %cmp = icmp slt i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_imm_i32:
; GCN: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8

; EG: MIN_INT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_imin_sle_imm_i32(i32 addrspace(1)* %out, i32 %a) #0 {
  %cmp = icmp sle i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ule_i32
; GCN: v_min_u32_e32

; EG: MIN_UINT
define amdgpu_kernel void @v_test_umin_ule_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ule_v3i32
; GCN: v_min_u32_e32
; GCN: v_min_u32_e32
; GCN: v_min_u32_e32
; GCN-NOT: v_min_u32_e32
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @v_test_umin_ule_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %a.ptr, <3 x i32> addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds <3 x i32>, <3 x i32> addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds <3 x i32>, <3 x i32> addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds <3 x i32>, <3 x i32> addrspace(1)* %out, i32 %tid

  %a = load <3 x i32>, <3 x i32> addrspace(1)* %a.gep
  %b = load <3 x i32>, <3 x i32> addrspace(1)* %b.gep
  %cmp = icmp ule <3 x i32> %a, %b
  %val = select <3 x i1> %cmp, <3 x i32> %a, <3 x i32> %b
  store <3 x i32> %val, <3 x i32> addrspace(1)* %out.gep
  ret void
}

; FIXME: Reduce unused packed component to scalar
; FUNC-LABEL: @v_test_umin_ule_v3i16{{$}}
; SI: v_min_u32_e32
; SI: v_min_u32_e32
; SI: v_min_u32_e32
; SI-NOT: v_min_u32_e32

; VI: v_min_u16_e32
; VI: v_min_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_min_u16_e32
; VI-NOT: v_min_u16

; GFX9_10: v_pk_min_u16
; GFX9_10: v_pk_min_u16

; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @v_test_umin_ule_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(1)* %a.ptr, <3 x i16> addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds <3 x i16>, <3 x i16> addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds <3 x i16>, <3 x i16> addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds <3 x i16>, <3 x i16> addrspace(1)* %out, i32 %tid

  %a = load <3 x i16>, <3 x i16> addrspace(1)* %a.gep
  %b = load <3 x i16>, <3 x i16> addrspace(1)* %b.gep
  %cmp = icmp ule <3 x i16> %a, %b
  %val = select <3 x i1> %cmp, <3 x i16> %a, <3 x i16> %b
  store <3 x i16> %val, <3 x i16> addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: @s_test_umin_ule_i32
; GCN: s_min_u32

; EG: MIN_UINT
define amdgpu_kernel void @s_test_umin_ule_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32
; GCN: v_min_u32_e32

; EG: MIN_UINT
define amdgpu_kernel void @v_test_umin_ult_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %a.ptr, i32 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i32, i32 addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds i32, i32 addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %a.gep, align 4
  %b = load i32, i32 addrspace(1)* %b.gep, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_umin_ult_i8:
; SI: {{buffer|flat|global}}_load_ubyte
; SI: {{buffer|flat|global}}_load_ubyte
; SI: v_min_u32_e32

; GFX8_9_10: {{flat|global}}_load_ubyte
; GFX8_9_10: {{flat|global}}_load_ubyte
; GFX8_9:    v_min_u16_e32
; GFX10:     v_min_u16_e64

; EG: MIN_UINT
define amdgpu_kernel void @v_test_umin_ult_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %a.ptr, i8 addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i8, i8 addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds i8, i8 addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds i8, i8 addrspace(1)* %out, i32 %tid

  %a = load i8, i8 addrspace(1)* %a.gep, align 1
  %b = load i8, i8 addrspace(1)* %b.gep, align 1
  %cmp = icmp ult i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out.gep, align 1
  ret void
}

; FUNC-LABEL: @s_test_umin_ult_i32
; GCN: s_min_u32

; EG: MIN_UINT
define amdgpu_kernel void @s_test_umin_ult_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32_multi_use
; SI-NOT: v_min
; GCN: v_cmp_lt_u32
; SI-NOT: v_min
; SI: v_cndmask_b32
; SI-NOT: v_min
; GCN: s_endpgm

; EG-NOT: MIN_UINT
define amdgpu_kernel void @v_test_umin_ult_i32_multi_use(i32 addrspace(1)* %out0, i1 addrspace(1)* %out1, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) #0 {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out0, align 4
  store i1 %cmp, i1 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i16_multi_use
; GCN-NOT: v_min
; GCN: v_cmp_lt_u32
; GCN: v_cndmask_b32
; GCN-NOT: v_min
; GCN: s_endpgm

; EG-NOT: MIN_UINT
define amdgpu_kernel void @v_test_umin_ult_i16_multi_use(i16 addrspace(1)* %out0, i1 addrspace(1)* %out1, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) #0 {
  %a = load i16, i16 addrspace(1)* %aptr, align 2
  %b = load i16, i16 addrspace(1)* %bptr, align 2
  %cmp = icmp ult i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out0, align 2
  store i1 %cmp, i1 addrspace(1)* %out1
  ret void
}


; FUNC-LABEL: @s_test_umin_ult_v1i32
; GCN: s_min_u32

; EG: MIN_UINT
define amdgpu_kernel void @s_test_umin_ult_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) #0 {
  %cmp = icmp ult <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i32:
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @s_test_umin_ult_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) #0 {
  %cmp = icmp ult <8 x i32> %a, %b
  %val = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %val, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i16:
; GCN-NOT: {{buffer|flat|global}}_load
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32

; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32
; VI: s_min_u32

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @s_test_umin_ult_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> %a, <8 x i16> %b) #0 {
  %cmp = icmp ult <8 x i16> %a, %b
  %val = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  store <8 x i16> %val, <8 x i16> addrspace(1)* %out
  ret void
}

; Make sure redundant and removed
; FUNC-LABEL: {{^}}simplify_demanded_bits_test_umin_ult_i16:
; GCN-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xa|0x28}}
; GCN-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0x13|0x4c}}
; GCN: s_min_u32 [[MIN:s[0-9]+]], [[A]], [[B]]
; GCN: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; GCN: buffer_store_dword [[VMIN]]

; EG: MIN_UINT
define amdgpu_kernel void @simplify_demanded_bits_test_umin_ult_i16(i32 addrspace(1)* %out, [8 x i32], i16 zeroext %a, [8 x i32], i16 zeroext %b) #0 {
  %a.ext = zext i16 %a to i32
  %b.ext = zext i16 %b to i32
  %cmp = icmp ult i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %mask = and i32 %val, 65535
  store i32 %mask, i32 addrspace(1)* %out
  ret void
}

; Make sure redundant sign_extend_inreg removed.

; FUNC-LABEL: {{^}}simplify_demanded_bits_test_min_slt_i16:
; GCN-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xa|0x28}}
; GCN-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0x13|0x4c}}
; GCN-DAG: s_sext_i32_i16 [[EXT_A:s[0-9]+]], [[A]]
; GCN-DAG: s_sext_i32_i16 [[EXT_B:s[0-9]+]], [[B]]

; GCN: s_min_i32 [[MIN:s[0-9]+]], [[EXT_A]], [[EXT_B]]
; GCN: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; GCN: buffer_store_dword [[VMIN]]

; EG: MIN_INT
define amdgpu_kernel void @simplify_demanded_bits_test_min_slt_i16(i32 addrspace(1)* %out, [8 x i32], i16 signext %a, [8 x i32], i16 signext %b) #0 {
  %a.ext = sext i16 %a to i32
  %b.ext = sext i16 %b to i32
  %cmp = icmp slt i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %shl = shl i32 %val, 16
  %sextinreg = ashr i32 %shl, 16
  store i32 %sextinreg, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i16:
; GCN: s_min_i32

; EG: MIN_INT
define amdgpu_kernel void @s_test_imin_sle_i16(i16 addrspace(1)* %out, i16 %a, i16 %b) #0 {
  %cmp = icmp sle i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}

; 64 bit
; FUNC-LABEL: {{^}}test_umin_ult_i64
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @test_umin_ult_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %tmp = icmp ult i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_umin_ule_i64
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @test_umin_ule_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %tmp = icmp ule i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imin_slt_i64
; GCN: s_endpgm

; EG-DAG: MIN_UINT
; EG-DAG: MIN_INT
define amdgpu_kernel void @test_imin_slt_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %tmp = icmp slt i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imin_sle_i64
; GCN: s_endpgm

; EG-DAG: MIN_UINT
; EG-DAG: MIN_INT
define amdgpu_kernel void @test_imin_sle_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %tmp = icmp sle i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_test_imin_sle_v2i16:
; SI: v_min_i32
; SI: v_min_i32

; VI: v_min_i16
; VI: v_min_i16

; GFX9_10: v_pk_min_i16

; EG: MIN_INT
; EG: MIN_INT
define amdgpu_kernel void @v_test_imin_sle_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %a.ptr, <2 x i16> addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %a.gep
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b.gep
  %cmp = icmp sle <2 x i16> %a, %b
  %val = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %val, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; FIXME: i16 min
; FUNC-LABEL: {{^}}v_test_imin_ule_v2i16:
; SI: v_min_u32
; SI: v_min_u32

; VI: v_min_u16
; VI: v_min_u16

; GFX9_10: v_pk_min_u16

; EG: MIN_UINT
; EG: MIN_UINT
define amdgpu_kernel void @v_test_imin_ule_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %a.ptr, <2 x i16> addrspace(1)* %b.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %a.ptr, i32 %tid
  %b.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %b.ptr, i32 %tid
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %a.gep
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b.gep
  %cmp = icmp ule <2 x i16> %a, %b
  %val = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %val, <2 x i16> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
