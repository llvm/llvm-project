; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=-flat-for-global,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)

; GCN-LABEL: {{^}}v_cnd_nan_nosgpr:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0
; GCN: s_cselect_b64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], -1, 0
; GCN: v_cndmask_b32_e{{32|64}} v{{[0-9]}}, -1, v{{[0-9]+}}, [[COND]]
; GCN-DAG: v{{[0-9]}}
; All nan values are converted to 0xffffffff
; GCN: s_endpgm
define amdgpu_kernel void @v_cnd_nan_nosgpr(ptr addrspace(1) %out, i32 %c, ptr addrspace(1) %fptr) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %f.gep = getelementptr float, ptr addrspace(1) %fptr, i32 %idx
  %f = load float, ptr addrspace(1) %f.gep
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, float 0xFFFFFFFFE0000000, float %f
  store float %select, ptr addrspace(1) %out
  ret void
}


; This requires slightly trickier SGPR operand legalization since the
; single constant bus SGPR usage is the last operand, and it should
; never be moved.
; However on GFX10 constant bus is limited to 2 scalar operands, not one.

; GCN-LABEL: {{^}}v_cnd_nan:
; SIVI:  s_cmp_eq_u32 s{{[0-9]+}}, 0
; SIVI:  s_cselect_b64 vcc, -1, 0
; SIVI:  v_cndmask_b32_e32 v{{[0-9]+}}, -1, v{{[0-9]+}}, vcc
; GFX10: s_cmp_eq_u32 s{{[0-9]+}}, 0
; GFX10: s_cselect_b64 [[CC:s\[[0-9:]+\]]],
; GFX10: v_cndmask_b32_e64 v{{[0-9]+}}, -1, s{{[0-9]+}}, [[CC]]
; GCN-DAG: v{{[0-9]}}
; All nan values are converted to 0xffffffff
; GCN: s_endpgm
define amdgpu_kernel void @v_cnd_nan(ptr addrspace(1) %out, i32 %c, float %f) #0 {
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, float 0xFFFFFFFFE0000000, float %f
  store float %select, ptr addrspace(1) %out
  ret void
}

; Test different compare and select operand types for optimal code
; shrinking.
; (select (cmp (sgprX, constant)), constant, sgprZ)

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k1_sgprZ_f32:
; GCN: s_load_{{dwordx2|b64}} s[[[X:[0-9]+]]:[[Z:[0-9]+]]], s[0:1], {{0x4c|0x13}}

; SIVI-DAG:  v_cmp_nlg_f32_e64 [[CC:vcc]], s[[X]], 0
; GFX10-DAG: v_cmp_nlg_f32_e64 [[CC:s\[[0-9:]+\]]], s[[X]], 0
; SIVI-DAG:  v_mov_b32_e32 [[VZ:v[0-9]+]], s[[Z]]
; SIVI:      v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[VZ]], [[CC]]
; GFX10:     v_cndmask_b32_e64 v{{[0-9]+}}, 1.0, s[[Z]], [[CC]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k1_sgprZ_f32(ptr addrspace(1) %out, [8 x i32], float %x, float %z) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 1.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k1_sgprX_f32:
; GCN: s_load_{{dword|b32}} [[X:s[0-9]+]]
; SIVI-DAG:  v_cmp_nlg_f32_e64 [[CC:vcc]], [[X]], 0
; GFX10-DAG: v_cmp_nlg_f32_e64 [[CC:s\[[0-9:]+\]]], [[X]], 0
; SIVI-DAG:  v_mov_b32_e32 [[VX:v[0-9]+]], [[X]]
; SIVI:      v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[VX]], [[CC]]
; GFX10:     v_cndmask_b32_e64 v{{[0-9]+}}, 1.0, [[X]], [[CC]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k1_sgprX_f32(ptr addrspace(1) %out, float %x) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 1.0, float %x
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k0_sgprZ_f32:
; GCN-DAG: s_load_{{dwordx2|b64}} s[[[X:[0-9]+]]:[[Z:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, {{0x13|0x4c}}
; SIVI-DAG:  v_cmp_nlg_f32_e64 [[CC:vcc]], s[[X]], 0
; GFX10-DAG: v_cmp_nlg_f32_e64 [[CC:s\[[0-9:]+\]]], s[[X]], 0
; SIVI-DAG:  v_mov_b32_e32 [[VZ:v[0-9]+]], s[[Z]]
; SIVI:      v_cndmask_b32_e32 v{{[0-9]+}}, 0, [[VZ]], [[CC]]
; GFX10:     v_cndmask_b32_e64 v{{[0-9]+}}, 0, s[[Z]], [[CC]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k0_sgprZ_f32(ptr addrspace(1) %out, [8 x i32], float %x, float %z) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 0.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k0_sgprX_f32:
; GCN: s_load_{{dword|b32}} [[X:s[0-9]+]]
; SIVI-DAG:  v_cmp_nlg_f32_e64 [[CC:vcc]], [[X]], 0
; GFX10-DAG: v_cmp_nlg_f32_e64 [[CC:s\[[0-9:]+\]]], [[X]], 0
; SIVI-DAG:  v_mov_b32_e32 [[VX:v[0-9]+]], [[X]]
; SIVI:      v_cndmask_b32_e32 v{{[0-9]+}}, 0, [[VX]], [[CC]]
; GFX10:     v_cndmask_b32_e64 v{{[0-9]+}}, 0, [[X]], [[CC]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k0_sgprX_f32(ptr addrspace(1) %out, float %x) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 0.0, float %x
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k0_vgprZ_f32:
; GCN-DAG: s_load_{{dword|b32}} [[X:s[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_{{dword|b32}} [[Z:v[0-9]+]]
; GCN-DAG: v_cmp_nlg_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], [[X]], 0
; GCN: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, 0, [[Z]], [[COND]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k0_vgprZ_f32(ptr addrspace(1) %out, float %x, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %z.gep = getelementptr inbounds float, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %z = load float, ptr addrspace(1) %z.gep
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 0.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_sgprX_k0_select_k1_vgprZ_f32:
; GCN-DAG: {{buffer|flat|global}}_load_{{dword|b32}} [[Z:v[0-9]+]]
; GCN-DAG: s_load_{{dword|b32}} [[X:s[0-9]+]]
; GCN-DAG: v_cmp_nlg_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], [[X]], 0
; GCN: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, 1.0, [[Z]], [[COND]]
define amdgpu_kernel void @fcmp_sgprX_k0_select_k1_vgprZ_f32(ptr addrspace(1) %out, float %x, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %z.gep = getelementptr inbounds float, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %z = load float, ptr addrspace(1) %z.gep
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, float 1.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_vgprX_k0_select_k1_sgprZ_f32:
; GCN-DAG: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN-DAG: s_load_{{dword|b32}} [[Z:s[0-9]+]]
; GCN-DAG: v_cmp_ngt_f32_e32 vcc, 0, [[X]]
; SIVI-DAG: v_mov_b32_e32 [[VZ:v[0-9]+]], [[Z]]
; SIVI:     v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[VZ]], vcc
; GFX10:    v_cndmask_b32_e64 v{{[0-9]+}}, 1.0, [[Z]], vcc
define amdgpu_kernel void @fcmp_vgprX_k0_select_k1_sgprZ_f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, float %z) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %x = load float, ptr addrspace(1) %x.gep
  %setcc = fcmp olt float %x, 0.0
  %select = select i1 %setcc, float 1.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_vgprX_k0_select_k1_vgprZ_f32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[Z:v[0-9]+]]
; GCN: v_cmp_le_f32_e32 vcc, 0, [[X]]
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[Z]], vcc
define amdgpu_kernel void @fcmp_vgprX_k0_select_k1_vgprZ_f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds float, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile float, ptr addrspace(1) %z.gep
  %setcc = fcmp ult float %x, 0.0
  %select = select i1 %setcc, float 1.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}icmp_vgprX_k0_select_k1_vgprZ_i32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[Z:v[0-9]+]]
; GCN: v_cmp_lt_i32_e32 vcc, -1, [[X]]
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 2, [[Z]], vcc
define amdgpu_kernel void @icmp_vgprX_k0_select_k1_vgprZ_i32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds i32, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds i32, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile i32, ptr addrspace(1) %x.gep
  %z = load volatile i32, ptr addrspace(1) %z.gep
  %setcc = icmp slt i32 %x, 0
  %select = select i1 %setcc, i32 2, i32 %z
  store i32 %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}icmp_vgprX_k0_select_k1_vgprZ_i64:
; GCN: {{buffer|flat|global}}_load_{{dwordx2|b64}} v[[[X_LO:[0-9]+]]:[[X_HI:[0-9]+]]]
; GCN-DAG: {{buffer|flat|global}}_load_{{dwordx2|b64}} v[[[Z_LO:[0-9]+]]:[[Z_HI:[0-9]+]]]
; GCN-DAG: v_cmp_lt_i64_e32 vcc, -1, v[[[X_LO]]:[[X_HI]]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v[[Z_HI]], vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 2, v[[Z_LO]], vcc
define amdgpu_kernel void @icmp_vgprX_k0_select_k1_vgprZ_i64(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds i64, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds i64, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile i64, ptr addrspace(1) %x.gep
  %z = load volatile i64, ptr addrspace(1) %z.gep
  %setcc = icmp slt i64 %x, 0
  %select = select i1 %setcc, i64 2, i64 %z
  store i64 %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_vgprX_k0_select_vgprZ_k1_v4f32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b128}}

; GCN: v_cmp_nge_f32_e32 vcc, 4.0, [[X]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, -0.5, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @fcmp_vgprX_k0_select_vgprZ_k1_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile <4 x float>, ptr addrspace(1) %z.gep
  %setcc = fcmp ugt float %x, 4.0
  %select = select i1 %setcc, <4 x float> %z, <4 x float> <float 1.0, float 2.0, float -0.5, float 4.0>
  store <4 x float> %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}fcmp_vgprX_k0_select_k1_vgprZ_v4f32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b128}}

; GCN: v_cmp_ge_f32_e32 vcc, 4.0, [[X]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, -0.5, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @fcmp_vgprX_k0_select_k1_vgprZ_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile <4 x float>, ptr addrspace(1) %z.gep
  %setcc = fcmp ugt float %x, 4.0
  %select = select i1 %setcc, <4 x float> <float 1.0, float 2.0, float -0.5, float 4.0>, <4 x float> %z
  store <4 x float> %select, ptr addrspace(1) %out.gep
  ret void
}

; This must be swapped as a vector type before the condition has
; multiple uses.

; GCN-LABEL: {{^}}fcmp_k0_vgprX_select_k1_vgprZ_v4f32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b128}}

; GCN: v_cmp_le_f32_e32 vcc, 4.0, [[X]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, -0.5, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @fcmp_k0_vgprX_select_k1_vgprZ_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile <4 x float>, ptr addrspace(1) %z.gep
  %setcc = fcmp ugt float 4.0, %x
  %select = select i1 %setcc, <4 x float> <float 1.0, float 2.0, float -0.5, float 4.0>, <4 x float> %z
  store <4 x float> %select, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}icmp_vgprX_k0_select_k1_vgprZ_i1:
; GCN: load_{{dword|b32}}
; GCN: load_{{ubyte|u8}}
; GCN-DAG: v_cmp_gt_i32_e32 vcc, 0, v
; GCN-DAG: v_and_b32_e32 v{{[0-9]+}}, 1,
; GCN-DAG: v_cmp_eq_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, 1, v
; GCN-DAG: s_or_b64 s{{\[[0-9]+:[0-9]+\]}}, vcc, s{{\[[0-9]+:[0-9]+\]}}
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, s
; GCN: store_{{byte|b8}}
define amdgpu_kernel void @icmp_vgprX_k0_select_k1_vgprZ_i1(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds i32, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds i1, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i1, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile i32, ptr addrspace(1) %x.gep
  %z = load volatile i1, ptr addrspace(1) %z.gep
  %setcc = icmp slt i32 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  store i1 %select, ptr addrspace(1) %out.gep
  ret void
}

; Different types compared vs. selected
; GCN-LABEL: {{^}}fcmp_vgprX_k0_selectf64_k1_vgprZ_f32:
; SIVI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3ff00000
; GCN-DAG: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_{{dwordx2|b64}}

; GCN: v_cmp_le_f32_e32 vcc, 0, [[X]]
; SIVI-DAG:  v_cndmask_b32_e32 v{{[0-9]+}}, [[K]], v{{[0-9]+}}, vcc
; GFX10-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 0x3ff00000, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @fcmp_vgprX_k0_selectf64_k1_vgprZ_f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds double, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile double, ptr addrspace(1) %z.gep
  %setcc = fcmp ult float %x, 0.0
  %select = select i1 %setcc, double 1.0, double %z
  store double %select, ptr addrspace(1) %out.gep
  ret void
}

; Different types compared vs. selected
; GCN-LABEL: {{^}}fcmp_vgprX_k0_selecti64_k1_vgprZ_f32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dwordx2|b64}}

; GCN: v_cmp_nlg_f32_e32 vcc, 0, [[X]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 3, v{{[0-9]+}}, vcc
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @fcmp_vgprX_k0_selecti64_k1_vgprZ_f32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds i64, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile i64, ptr addrspace(1) %z.gep
  %setcc = fcmp one float %x, 0.0
  %select = select i1 %setcc, i64 3, i64 %z
  store i64 %select, ptr addrspace(1) %out.gep
  ret void
}

; Different types compared vs. selected
; GCN-LABEL: {{^}}icmp_vgprX_k0_selectf32_k1_vgprZ_i32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[Z:v[0-9]+]]

; GCN: v_cmp_gt_u32_e32 vcc, 2, [[X]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 4.0, [[Z]], vcc
define amdgpu_kernel void @icmp_vgprX_k0_selectf32_k1_vgprZ_i32(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds i32, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds float, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile i32, ptr addrspace(1) %x.gep
  %z = load volatile float, ptr addrspace(1) %z.gep
  %setcc = icmp ugt i32 %x, 1
  %select = select i1 %setcc, float 4.0, float %z
  store float %select, ptr addrspace(1) %out.gep
  ret void
}

; FIXME: Should be able to handle multiple uses

; GCN-LABEL: {{^}}fcmp_k0_vgprX_select_k1_vgprZ_f32_cond_use_x2:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[X:v[0-9]+]]

; GCN: v_cmp_nle_f32_e32 vcc, 4.0, [[X]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, -1.0, vcc
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, -2.0, vcc
define amdgpu_kernel void @fcmp_k0_vgprX_select_k1_vgprZ_f32_cond_use_x2(ptr addrspace(1) %out, ptr addrspace(1) %x.ptr, ptr addrspace(1) %z.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %x.gep = getelementptr inbounds float, ptr addrspace(1) %x.ptr, i64 %tid.ext
  %z.gep = getelementptr inbounds float, ptr addrspace(1) %z.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i64 %tid.ext
  %x = load volatile float, ptr addrspace(1) %x.gep
  %z = load volatile float, ptr addrspace(1) %z.gep
  %setcc = fcmp ugt float 4.0, %x
  %select0 = select i1 %setcc, float -1.0, float %z
  %select1 = select i1 %setcc, float -2.0, float %z
  store volatile float %select0, ptr addrspace(1) %out.gep
  store volatile float %select1, ptr addrspace(1) %out.gep
  ret void
}

; Source modifiers abs/neg only work for f32

; GCN-LABEL: {{^}}v_cndmask_abs_neg_f16:
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}},
define amdgpu_kernel void @v_cndmask_abs_neg_f16(ptr addrspace(1) %out, i32 %c, ptr addrspace(1) %fptr) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %f.gep = getelementptr half, ptr addrspace(1) %fptr, i32 %idx
  %f = load half, ptr addrspace(1) %f.gep
  %f.abs = call half @llvm.fabs.f16(half %f)
  %f.neg = fneg half %f
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, half %f.abs, half %f.neg
  store half %select, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_cndmask_abs_neg_f32:
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, -v{{[0-9]+}}, |v{{[0-9]+}}|,
define amdgpu_kernel void @v_cndmask_abs_neg_f32(ptr addrspace(1) %out, i32 %c, ptr addrspace(1) %fptr) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %f.gep = getelementptr float, ptr addrspace(1) %fptr, i32 %idx
  %f = load float, ptr addrspace(1) %f.gep
  %f.abs = call float @llvm.fabs.f32(float %f)
  %f.neg = fneg float %f
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, float %f.abs, float %f.neg
  store float %select, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_cndmask_abs_neg_f64:
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}},
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}},
define amdgpu_kernel void @v_cndmask_abs_neg_f64(ptr addrspace(1) %out, i32 %c, ptr addrspace(1) %fptr) #0 {
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %f.gep = getelementptr double, ptr addrspace(1) %fptr, i32 %idx
  %f = load double, ptr addrspace(1) %f.gep
  %f.abs = call double @llvm.fabs.f64(double %f)
  %f.neg = fneg double %f
  %setcc = icmp ne i32 %c, 0
  %select = select i1 %setcc, double %f.abs, double %f.neg
  store double %select, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
