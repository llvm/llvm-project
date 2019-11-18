; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -amdgpu-early-ifcvt=1 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -amdgpu-early-ifcvt=1 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032,GFX10DEFWAVE %s

; GCN-LABEL: {{^}}test_vopc_i32:
; GFX1032: v_cmp_lt_i32_e32 vcc_lo, 0, v{{[0-9]+}}
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, vcc_lo
; GFX1064: v_cmp_lt_i32_e32 vcc, 0, v{{[0-9]+}}
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, vcc{{$}}
define amdgpu_kernel void @test_vopc_i32(i32 addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %lid
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp sgt i32 %load, 0
  %sel = select i1 %cmp, i32 1, i32 2
  store i32 %sel, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vopc_f32:
; GFX1032: v_cmp_nge_f32_e32 vcc_lo, 0, v{{[0-9]+}}
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 2.0, 1.0, vcc_lo
; GFX1064: v_cmp_nge_f32_e32 vcc, 0, v{{[0-9]+}}
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 2.0, 1.0, vcc{{$}}
define amdgpu_kernel void @test_vopc_f32(float addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %lid
  %load = load float, float addrspace(1)* %gep, align 4
  %cmp = fcmp ugt float %load, 0.0
  %sel = select i1 %cmp, float 1.0, float 2.0
  store float %sel, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vopc_vcmpx:
; GFX1032: v_cmpx_le_f32_e32 0, v{{[0-9]+}}
; GFX1064: v_cmpx_le_f32_e32 0, v{{[0-9]+}}
define amdgpu_ps void @test_vopc_vcmpx(float %x) {
  %cmp = fcmp oge float %x, 0.0
  call void @llvm.amdgcn.kill(i1 %cmp)
  ret void
}

; GCN-LABEL: {{^}}test_vopc_2xf16:
; GFX1032: v_cmp_le_f16_sdwa [[SC:s[0-9]+]], {{[vs][0-9]+}}, v{{[0-9]+}} src0_sel:WORD_1 src1_sel:DWORD
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 0x3c003c00, v{{[0-9]+}}, [[SC]]
; GFX1064: v_cmp_le_f16_sdwa [[SC:s\[[0-9:]+\]]], {{[vs][0-9]+}}, v{{[0-9]+}} src0_sel:WORD_1 src1_sel:DWORD
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 0x3c003c00, v{{[0-9]+}}, [[SC]]
define amdgpu_kernel void @test_vopc_2xf16(<2 x half> addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i32 %lid
  %load = load <2 x half>, <2 x half> addrspace(1)* %gep, align 4
  %elt = extractelement <2 x half> %load, i32 1
  %cmp = fcmp ugt half %elt, 0.0
  %sel = select i1 %cmp, <2 x half> <half 1.0, half 1.0>, <2 x half> %load
  store <2 x half> %sel, <2 x half> addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vopc_class:
; GFX1032: v_cmp_class_f32_e64 [[C:vcc_lo|s[0-9:]+]], s{{[0-9]+}}, 0x204
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[C]]
; GFX1064: v_cmp_class_f32_e64 [[C:vcc|s\[[0-9:]+\]]], s{{[0-9]+}}, 0x204
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[C]]{{$}}
define amdgpu_kernel void @test_vopc_class(i32 addrspace(1)* %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x)
  %cmp = fcmp oeq float %fabs, 0x7FF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vcmp_vcnd_f16:
; GFX1032: v_cmp_neq_f16_e64 [[C:vcc_lo|s\[[0-9:]+\]]], 0x7c00, s{{[0-9]+}}
; GFX1032: v_cndmask_b32_e32 v{{[0-9]+}}, 0x3c00, v{{[0-9]+}}, [[C]]

; GFX1064: v_cmp_neq_f16_e64 [[C:vcc|s\[[0-9:]+\]]], 0x7c00, s{{[0-9]+}}
; GFX1064: v_cndmask_b32_e32 v{{[0-9]+}}, 0x3c00, v{{[0-9]+}}, [[C]]{{$}}
define amdgpu_kernel void @test_vcmp_vcnd_f16(half addrspace(1)* %out, half %x) #0 {
  %cmp = fcmp oeq half %x, 0x7FF0000000000000
  %sel = select i1 %cmp, half 1.0, half %x
  store half %sel, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}test_vop3_cmp_f32_sop_and:
; GFX1032: v_cmp_nge_f32_e32 vcc_lo, 0, v{{[0-9]+}}
; GFX1032: v_cmp_nle_f32_e64 [[C2:s[0-9]+]], 1.0, v{{[0-9]+}}
; GFX1032: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[C2]]
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 2.0, 1.0, [[AND]]
; GFX1064: v_cmp_nge_f32_e32 vcc, 0, v{{[0-9]+}}
; GFX1064: v_cmp_nle_f32_e64 [[C2:s\[[0-9:]+\]]], 1.0, v{{[0-9]+}}
; GFX1064: s_and_b64 [[AND:s\[[0-9:]+\]]], vcc, [[C2]]
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 2.0, 1.0, [[AND]]
define amdgpu_kernel void @test_vop3_cmp_f32_sop_and(float addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %lid
  %load = load float, float addrspace(1)* %gep, align 4
  %cmp = fcmp ugt float %load, 0.0
  %cmp2 = fcmp ult float %load, 1.0
  %and = and i1 %cmp, %cmp2
  %sel = select i1 %and, float 1.0, float 2.0
  store float %sel, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vop3_cmp_i32_sop_xor:
; GFX1032: v_cmp_lt_i32_e32 vcc_lo, 0, v{{[0-9]+}}
; GFX1032: v_cmp_gt_i32_e64 [[C2:s[0-9]+]], 1, v{{[0-9]+}}
; GFX1032: s_xor_b32 [[AND:s[0-9]+]], vcc_lo, [[C2]]
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, [[AND]]
; GFX1064: v_cmp_lt_i32_e32 vcc, 0, v{{[0-9]+}}
; GFX1064: v_cmp_gt_i32_e64 [[C2:s\[[0-9:]+\]]], 1, v{{[0-9]+}}
; GFX1064: s_xor_b64 [[AND:s\[[0-9:]+\]]], vcc, [[C2]]
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, [[AND]]
define amdgpu_kernel void @test_vop3_cmp_i32_sop_xor(i32 addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %lid
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp sgt i32 %load, 0
  %cmp2 = icmp slt i32 %load, 1
  %xor = xor i1 %cmp, %cmp2
  %sel = select i1 %xor, i32 1, i32 2
  store i32 %sel, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_vop3_cmp_u32_sop_or:
; GFX1032: v_cmp_lt_u32_e32 vcc_lo, 3, v{{[0-9]+}}
; GFX1032: v_cmp_gt_u32_e64 [[C2:s[0-9]+]], 2, v{{[0-9]+}}
; GFX1032: s_or_b32 [[AND:s[0-9]+]], vcc_lo, [[C2]]
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, [[AND]]
; GFX1064: v_cmp_lt_u32_e32 vcc, 3, v{{[0-9]+}}
; GFX1064: v_cmp_gt_u32_e64 [[C2:s\[[0-9:]+\]]], 2, v{{[0-9]+}}
; GFX1064: s_or_b64 [[AND:s\[[0-9:]+\]]], vcc, [[C2]]
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 2, 1, [[AND]]
define amdgpu_kernel void @test_vop3_cmp_u32_sop_or(i32 addrspace(1)* %arg) {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %lid
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %load, 3
  %cmp2 = icmp ult i32 %load, 2
  %or = or i1 %cmp, %cmp2
  %sel = select i1 %or, i32 1, i32 2
  store i32 %sel, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_mask_if:
; GFX1032: s_and_saveexec_b32 s{{[0-9]+}}, vcc_lo
; GFX1064: s_and_saveexec_b64 s[{{[0-9:]+}}], vcc{{$}}
; GCN: ; mask branch
define amdgpu_kernel void @test_mask_if(i32 addrspace(1)* %arg) #0 {
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp ugt i32 %lid, 10
  br i1 %cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %arg, align 4
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}test_loop_with_if:
; GFX1032: s_or_b32 s{{[0-9]+}}, vcc_lo, s{{[0-9]+}}
; GFX1032: s_andn2_b32 exec_lo, exec_lo, s{{[0-9]+}}
; GFX1064: s_or_b64 s[{{[0-9:]+}}], vcc, s[{{[0-9:]+}}]
; GFX1064: s_andn2_b64 exec, exec, s[{{[0-9:]+}}]
; GCN:     s_cbranch_execz
; GCN:   BB{{.*}}:
; GFX1032: s_and_saveexec_b32 s{{[0-9]+}}, vcc_lo
; GFX1064: s_and_saveexec_b64 s[{{[0-9:]+}}], vcc{{$}}
; GCN:     s_cbranch_execz
; GCN:   BB{{.*}}:
; GCN:   BB{{.*}}:
; GFX1032: s_xor_b32 s{{[0-9]+}}, exec_lo, s{{[0-9]+}}
; GFX1064: s_xor_b64 s[{{[0-9:]+}}], exec, s[{{[0-9:]+}}]
; GCN:     ; mask branch BB
; GCN:   BB{{.*}}:
; GCN:   BB{{.*}}:
; GFX1032: s_or_b32 exec_lo, exec_lo, s{{[0-9]+}}
; GFX1032: s_and_saveexec_b32 s{{[0-9]+}}, s{{[0-9]+}}
; GFX1064: s_or_b64 exec, exec, s[{{[0-9:]+}}]
; GFX1064: s_and_saveexec_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
; GCN:     ; mask branch BB
; GCN:   BB{{.*}}:
; GCN:   BB{{.*}}:
; GCN:     s_endpgm
define amdgpu_kernel void @test_loop_with_if(i32 addrspace(1)* %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb2

bb1:
  ret void

bb2:
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp15, %bb13 ]
  %tmp4 = icmp slt i32 %tmp3, %tmp
  br i1 %tmp4, label %bb5, label %bb11

bb5:
  %tmp6 = sext i32 %tmp3 to i64
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp6
  %tmp8 = load i32, i32 addrspace(1)* %tmp7, align 4
  %tmp9 = icmp sgt i32 %tmp8, 10
  br i1 %tmp9, label %bb10, label %bb11

bb10:
  store i32 %tmp, i32 addrspace(1)* %tmp7, align 4
  br label %bb13

bb11:
  %tmp12 = sdiv i32 %tmp3, 2
  br label %bb13

bb13:
  %tmp14 = phi i32 [ %tmp3, %bb10 ], [ %tmp12, %bb11 ]
  %tmp15 = add nsw i32 %tmp14, 1
  %tmp16 = icmp slt i32 %tmp14, 255
  br i1 %tmp16, label %bb2, label %bb1
}

; GCN-LABEL: {{^}}test_loop_with_if_else_break:
; GFX1032: s_and_saveexec_b32 s{{[0-9]+}}, vcc_lo
; GFX1064: s_and_saveexec_b64 s[{{[0-9:]+}}], vcc{{$}}
; GCN:     ; mask branch
; GCN:     s_cbranch_execz
; GCN:   BB{{.*}}:
; GCN:   BB{{.*}}:

; GFX1032: s_or_b32 [[MASK0:s[0-9]+]], [[MASK0]], vcc_lo
; GFX1064: s_or_b64 [[MASK0:s\[[0-9:]+\]]], [[MASK0]], vcc
; GFX1032: s_andn2_b32 [[MASK1:s[0-9]+]], [[MASK1]], exec_lo
; GFX1064: s_andn2_b64 [[MASK1:s\[[0-9:]+\]]], [[MASK1]], exec
; GCN:     global_store_dword
; GFX1032: s_and_b32 [[MASK0]], [[MASK0]], exec_lo
; GFX1064: s_and_b64 [[MASK0]], [[MASK0]], exec
; GFX1032: s_or_b32 [[MASK1]], [[MASK1]], [[MASK0]]
; GFX1064: s_or_b64 [[MASK1]], [[MASK1]], [[MASK0]]
; GCN:   BB{{.*}}: ; %Flow
; GFX1032: s_and_b32 [[TMP0:s[0-9]+]], exec_lo, [[MASK1]]
; GFX1064: s_and_b64 [[TMP0:s\[[0-9:]+\]]], exec, [[MASK1]]
; GFX1032: s_or_b32  [[ACC:s[0-9]+]], [[TMP0]], [[ACC]]
; GFX1064: s_or_b64  [[ACC:s\[[0-9:]+\]]], [[TMP0]], [[ACC]]
; GFX1032: s_andn2_b32 exec_lo, exec_lo, [[ACC]]
; GFX1064: s_andn2_b64 exec, exec, [[ACC]]
; GCN:     s_cbranch_execz
; GCN:   BB{{.*}}:
; GCN: s_load_dword [[LOAD:s[0-9]+]]
; GFX1032: s_or_b32 [[MASK1]], [[MASK1]], exec_lo
; GFX1064: s_or_b64 [[MASK1]], [[MASK1]], exec
; GCN: s_cmp_lt_i32 [[LOAD]], 11
define amdgpu_kernel void @test_loop_with_if_else_break(i32 addrspace(1)* %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = icmp eq i32 %tmp, 0
  br i1 %tmp1, label %.loopexit, label %.preheader

.preheader:
  br label %bb2

bb2:
  %tmp3 = phi i32 [ %tmp9, %bb8 ], [ 0, %.preheader ]
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp4
  %tmp6 = load i32, i32 addrspace(1)* %tmp5, align 4
  %tmp7 = icmp sgt i32 %tmp6, 10
  br i1 %tmp7, label %bb8, label %.loopexit

bb8:
  store i32 %tmp, i32 addrspace(1)* %tmp5, align 4
  %tmp9 = add nuw nsw i32 %tmp3, 1
  %tmp10 = icmp ult i32 %tmp9, 256
  %tmp11 = icmp ult i32 %tmp9, %tmp
  %tmp12 = and i1 %tmp10, %tmp11
  br i1 %tmp12, label %bb2, label %.loopexit

.loopexit:
  ret void
}

; GCN-LABEL: {{^}}test_addc_vop2b:
; GFX1032: v_add_co_u32_e64 v{{[0-9]+}}, vcc_lo, v{{[0-9]+}}, s{{[0-9]+}}
; GFX1032: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, s{{[0-9]+}}, v{{[0-9]+}}, vcc_lo
; GFX1064: v_add_co_u32_e64 v{{[0-9]+}}, vcc, v{{[0-9]+}}, s{{[0-9]+}}
; GFX1064: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}, vcc{{$}}
define amdgpu_kernel void @test_addc_vop2b(i64 addrspace(1)* %arg, i64 %arg1) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %tmp
  %tmp4 = load i64, i64 addrspace(1)* %tmp3, align 8
  %tmp5 = add nsw i64 %tmp4, %arg1
  store i64 %tmp5, i64 addrspace(1)* %tmp3, align 8
  ret void
}

; GCN-LABEL: {{^}}test_subbrev_vop2b:
; GFX1032: v_sub_co_u32_e64 v{{[0-9]+}}, [[A0:s[0-9]+|vcc_lo]], v{{[0-9]+}}, s{{[0-9]+}}{{$}}
; GFX1032: v_subrev_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[A0]]{{$}}
; GFX1064: v_sub_co_u32_e64 v{{[0-9]+}}, [[A0:s\[[0-9:]+\]|vcc]], v{{[0-9]+}}, s{{[0-9]+}}{{$}}
; GFX1064: v_subrev_co_ci_u32_e32 v{{[0-9]+}}, vcc, {{[vs][0-9]+}}, {{[vs][0-9]+}}, [[A0]]{{$}}
define amdgpu_kernel void @test_subbrev_vop2b(i64 addrspace(1)* %arg, i64 %arg1) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %tmp
  %tmp4 = load i64, i64 addrspace(1)* %tmp3, align 8
  %tmp5 = sub nsw i64 %tmp4, %arg1
  store i64 %tmp5, i64 addrspace(1)* %tmp3, align 8
  ret void
}

; GCN-LABEL: {{^}}test_subb_vop2b:
; GFX1032: v_sub_co_u32_e64 v{{[0-9]+}}, [[A0:s[0-9]+|vcc_lo]], s{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX1032: v_sub_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, {{[vs][0-9]+}}, v{{[0-9]+}}, [[A0]]{{$}}
; GFX1064: v_sub_co_u32_e64 v{{[0-9]+}}, [[A0:s\[[0-9:]+\]|vcc]], s{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX1064: v_sub_co_ci_u32_e32 v{{[0-9]+}}, vcc, {{[vs][0-9]+}}, v{{[0-9]+}}, [[A0]]{{$}}
define amdgpu_kernel void @test_subb_vop2b(i64 addrspace(1)* %arg, i64 %arg1) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %tmp
  %tmp4 = load i64, i64 addrspace(1)* %tmp3, align 8
  %tmp5 = sub nsw i64 %arg1, %tmp4
  store i64 %tmp5, i64 addrspace(1)* %tmp3, align 8
  ret void
}

; GCN-LABEL: {{^}}test_udiv64:
; GFX1032: v_add_co_u32_e64 v{{[0-9]+}}, [[SDST:s[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX1032: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, 0, v{{[0-9]+}}, vcc_lo
; GFX1032: v_add_co_ci_u32_e64 v{{[0-9]+}}, vcc_lo, v{{[0-9]+}}, v{{[0-9]+}}, [[SDST]]
; GFX1032: v_add_co_u32_e64 v{{[0-9]+}}, vcc_lo, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1032: v_add_co_u32_e64 v{{[0-9]+}}, vcc_lo, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1032: v_add_co_u32_e64 v{{[0-9]+}}, vcc_lo, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1032: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, 0, v{{[0-9]+}}, vcc_lo
; GFX1032: v_sub_co_u32_e64 v{{[0-9]+}}, vcc_lo, s{{[0-9]+}}, v{{[0-9]+}}
; GFX1032: v_sub_co_ci_u32_e64 v{{[0-9]+}}, s{{[0-9]+}}, {{[vs][0-9]+}}, v{{[0-9]+}}, vcc_lo
; GFX1032: v_subrev_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, {{[vs][0-9]+}}, v{{[0-9]+}}, vcc_lo
; GFX1064: v_add_co_u32_e64 v{{[0-9]+}}, [[SDST:s\[[0-9:]+\]]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; GFX1064: v_add_co_ci_u32_e64 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}, [[SDST]]
; GFX1064: v_add_co_u32_e64 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_add_co_u32_e64 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_add_co_u32_e64 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc{{$}}
; GFX1064: v_sub_co_u32_e64 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_sub_co_ci_u32_e64 v{{[0-9]+}}, s[{{[0-9:]+}}], {{[vs][0-9]+}}, v{{[0-9]+}}, vcc{{$}}
; GFX1064: v_subrev_co_ci_u32_e32 v{{[0-9]+}}, vcc, {{[vs][0-9]+}}, v{{[0-9]+}}, vcc{{$}}
define amdgpu_kernel void @test_udiv64(i64 addrspace(1)* %arg) #0 {
bb:
  %tmp = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 1
  %tmp1 = load i64, i64 addrspace(1)* %tmp, align 8
  %tmp2 = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = udiv i64 %tmp1, %tmp2
  %tmp4 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 2
  store i64 %tmp3, i64 addrspace(1)* %tmp4, align 8
  ret void
}

; GCN-LABEL: {{^}}test_div_scale_f32:
; GFX1032: v_div_scale_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX1064: v_div_scale_f32 v{{[0-9]+}}, s[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_div_scale_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_scale_f64:
; GFX1032: v_div_scale_f64 v[{{[0-9:]+}}], s{{[0-9]+}}, v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX1064: v_div_scale_f64 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @test_div_scale_f64(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0, align 8
  %b = load volatile double, double addrspace(1)* %gep.1, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_mad_i64_i32:
; GFX1032: v_mad_i64_i32 v[{{[0-9:]+}}], s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX1064: v_mad_i64_i32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9:]+}}]
define i64 @test_mad_i64_i32(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = sext i32 %arg0 to i64
  %sext1 = sext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %mul, %arg2
  ret i64 %mad
}

; GCN-LABEL: {{^}}test_mad_u64_u32:
; GFX1032: v_mad_u64_u32 v[{{[0-9:]+}}], s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX1064: v_mad_u64_u32 v[{{[0-9:]+}}], s[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9:]+}}]
define i64 @test_mad_u64_u32(i32 %arg0, i32 %arg1, i64 %arg2) #0 {
  %sext0 = zext i32 %arg0 to i64
  %sext1 = zext i32 %arg1 to i64
  %mul = mul i64 %sext0, %sext1
  %mad = add i64 %mul, %arg2
  ret i64 %mad
}

; GCN-LABEL: {{^}}test_div_fmas_f32:
; GFX1032: v_cmp_eq_u32_e64 vcc_lo,
; GFX1064: v_cmp_eq_u32_e64 vcc,
; GCN:     v_div_fmas_f32 v{{[0-9]+}}, {{[vs][0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_div_fmas_f32(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f64:
; GFX1032: v_cmp_eq_u32_e64 vcc_lo,
; GFX1064: v_cmp_eq_u32_e64 vcc,
; GCN-DAG: v_div_fmas_f64 v[{{[0-9:]+}}], {{[vs]}}[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @test_div_fmas_f64(double addrspace(1)* %out, double %a, double %b, double %c, i1 %d) nounwind {
  %result = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 %d) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_i1_phi_vcc:
; GFX1032: s_mov_b32 [[VCC:vcc_lo]], 0{{$}}
; GFX1064: s_mov_b64 [[VCC:vcc]], 0{{$}}
; GFX1032: s_and_saveexec_b32 [[SAVE:s[0-9]+]], s{{[0-9]+}}{{$}}
; GFX1064: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], s[{{[0-9:]+}}]{{$}}

; GCN: load_dword [[LOAD:v[0-9]+]]
; GCN: v_cmp_ne_u32_e32 [[VCC]], 0, [[LOAD]]

; GCN: BB{{[0-9_]+}}:
; GFX1032: s_or_b32 exec_lo, exec_lo, [[SAVE]]
; GFX1064: s_or_b64 exec, exec, [[SAVE]]
; GCN: v_div_fmas_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @test_div_fmas_f32_i1_phi_vcc(float addrspace(1)* %out, float addrspace(1)* %in, i32 addrspace(1)* %dummy) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 2
  %gep.a = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.b = getelementptr float, float addrspace(1)* %gep.a, i32 1
  %gep.c = getelementptr float, float addrspace(1)* %gep.a, i32 2

  %a = load float, float addrspace(1)* %gep.a
  %b = load float, float addrspace(1)* %gep.b
  %c = load float, float addrspace(1)* %gep.c

  %cmp0 = icmp eq i32 %tid, 0
  br i1 %cmp0, label %bb, label %exit

bb:
  %val = load volatile i32, i32 addrspace(1)* %dummy
  %cmp1 = icmp ne i32 %val, 0
  br label %exit

exit:
  %cond = phi i1 [false, %entry], [%cmp1, %bb]
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %cond) nounwind readnone
  store float %result, float addrspace(1)* %gep.out, align 4
  ret void
}

; GCN-LABEL: {{^}}fdiv_f32:
; GFC1032: v_div_scale_f32 v{{[0-9]+}}, vcc_lo, s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
; GFC1064: v_div_scale_f32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_rcp_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NOT: vcc
; GCN: v_div_fmas_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @fdiv_f32(float addrspace(1)* %out, float %a, float %b) #0 {
entry:
  %fdiv = fdiv float %a, %b
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_br_cc_f16:
; GFX1032:      v_cmp_nlt_f16_e32 vcc_lo,
; GFX1032-NEXT: s_and_b32 vcc_lo, exec_lo, vcc_lo
; GFX1064:      v_cmp_nlt_f16_e32 vcc,
; GFX1064-NEXT: s_and_b64 vcc, exec, vcc{{$}}
; GCN-NEXT: s_cbranch_vccnz
define amdgpu_kernel void @test_br_cc_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %fcmp = fcmp olt half %a.val, %b.val
  br i1 %fcmp, label %one, label %two

one:
  store half %a.val, half addrspace(1)* %r
  ret void

two:
  store half %b.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}test_brcc_i1:
; GCN:      s_cmp_eq_u32 s{{[0-9]+}}, 0
; GCN-NEXT: s_cbranch_scc1
define amdgpu_kernel void @test_brcc_i1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i1 %val) #0 {
  %cmp0 = icmp ne i1 %val, 0
  br i1 %cmp0, label %store, label %end

store:
  store i32 222, i32 addrspace(1)* %out
  ret void

end:
  ret void
}

; GCN-LABEL: {{^}}test_preserve_condition_undef_flag:
; GFX1032: v_cmp_nlt_f32_e64 s{{[0-9]+}}, s{{[0-9]+}}, 1.0
; GFX1032: v_cmp_ngt_f32_e64 s{{[0-9]+}}, s{{[0-9]+}}, 0
; GFX1032: v_cmp_nlt_f32_e64 s{{[0-9]+}}, s{{[0-9]+}}, 1.0
; GFX1032: s_or_b32 [[OR1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX1032: s_or_b32 [[OR2:s[0-9]+]], [[OR1]], s{{[0-9]+}}
; GFX1032: s_and_b32 vcc_lo, exec_lo, [[OR2]]
; GFX1064: v_cmp_nlt_f32_e64 s[{{[0-9:]+}}], s{{[0-9]+}}, 1.0
; GFX1064: v_cmp_ngt_f32_e64 s[{{[0-9:]+}}], s{{[0-9]+}}, 0
; GFX1064: v_cmp_nlt_f32_e64 s[{{[0-9:]+}}], s{{[0-9]+}}, 1.0
; GFX1064: s_or_b64 [[OR1:s\[[0-9:]+\]]], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GFX1064: s_or_b64 [[OR2:s\[[0-9:]+\]]], [[OR1]], s[{{[0-9:]+}}]
; GFX1064: s_and_b64 vcc, exec, [[OR2]]
; GCN:     s_cbranch_vccnz
define amdgpu_kernel void @test_preserve_condition_undef_flag(float %arg, i32 %arg1, float %arg2) #0 {
bb0:
  %tmp = icmp sgt i32 %arg1, 4
  %undef = call i1 @llvm.amdgcn.class.f32(float undef, i32 undef)
  %tmp4 = select i1 %undef, float %arg, float 1.000000e+00
  %tmp5 = fcmp ogt float %arg2, 0.000000e+00
  %tmp6 = fcmp olt float %arg2, 1.000000e+00
  %tmp7 = fcmp olt float %arg, %tmp4
  %tmp8 = and i1 %tmp5, %tmp6
  %tmp9 = and i1 %tmp8, %tmp7
  br i1 %tmp9, label %bb1, label %bb2

bb1:
  store volatile i32 0, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}

; GCN-LABEL: {{^}}test_invert_true_phi_cond_break_loop:
; GFX1032: s_xor_b32 s{{[0-9]+}}, s{{[0-9]+}}, -1
; GFX1032: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX1064: s_xor_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], -1
; GFX1064: s_or_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_invert_true_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
  br i1 %tmp3, label %bb1, label %bb9

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}test_movrels_extract_neg_offset_vgpr:
; GFX1032: v_cmp_eq_u32_e32 vcc_lo, 1, v{{[0-9]+}}
; GFX1032: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc_lo
; GFX1032: v_cmp_ne_u32_e32 vcc_lo, 2, v{{[0-9]+}}
; GFX1032: v_cndmask_b32_e32 v{{[0-9]+}}, 2, v{{[0-9]+}}, vcc_lo
; GFX1032: v_cmp_ne_u32_e32 vcc_lo, 3, v{{[0-9]+}}
; GFX1032: v_cndmask_b32_e32 v{{[0-9]+}}, 3, v{{[0-9]+}}, vcc_lo
; GFX1064: v_cmp_eq_u32_e32 vcc, 1, v{{[0-9]+}}
; GFX1064: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
; GFX1064: v_cmp_ne_u32_e32 vcc, 2, v{{[0-9]+}}
; GFX1064: v_cndmask_b32_e32 v{{[0-9]+}}, 2, v{{[0-9]+}}, vcc
; GFX1064: v_cmp_ne_u32_e32 vcc, 3, v{{[0-9]+}}
; GFX1064: v_cndmask_b32_e32 v{{[0-9]+}}, 3, v{{[0-9]+}}, vcc
define amdgpu_kernel void @test_movrels_extract_neg_offset_vgpr(i32 addrspace(1)* %out) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -512
  %value = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_set_inactive:
; GFX1032: s_not_b32 exec_lo, exec_lo
; GFX1032: v_mov_b32_e32 {{v[0-9]+}}, 42
; GFX1032: s_not_b32 exec_lo, exec_lo
; GFX1064: s_not_b64 exec, exec{{$}}
; GFX1064: v_mov_b32_e32 {{v[0-9]+}}, 42
; GFX1064: s_not_b64 exec, exec{{$}}
define amdgpu_kernel void @test_set_inactive(i32 addrspace(1)* %out, i32 %in) #0 {
  %tmp = call i32 @llvm.amdgcn.set.inactive.i32(i32 %in, i32 42)
  store i32 %tmp, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_set_inactive_64:
; GFX1032: s_not_b32 exec_lo, exec_lo
; GFX1032: v_mov_b32_e32 {{v[0-9]+}}, 0
; GFX1032: v_mov_b32_e32 {{v[0-9]+}}, 0
; GFX1032: s_not_b32 exec_lo, exec_lo
; GFX1064: s_not_b64 exec, exec{{$}}
; GFX1064: v_mov_b32_e32 {{v[0-9]+}}, 0
; GFX1064: v_mov_b32_e32 {{v[0-9]+}}, 0
; GFX1064: s_not_b64 exec, exec{{$}}
define amdgpu_kernel void @test_set_inactive_64(i64 addrspace(1)* %out, i64 %in) #0 {
  %tmp = call i64 @llvm.amdgcn.set.inactive.i64(i64 %in, i64 0)
  store i64 %tmp, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_kill_i1_terminator_float:
; GFX1032: s_mov_b32 exec_lo, 0
; GFX1064: s_mov_b64 exec, 0
define amdgpu_ps void @test_kill_i1_terminator_float() #0 {
  call void @llvm.amdgcn.kill(i1 false)
  ret void
}

; GCN-LABEL: {{^}}test_kill_i1_terminator_i1:
; GFX1032: s_or_b32 [[OR:s[0-9]+]],
; GFX1032: s_and_b32 exec_lo, exec_lo, [[OR]]
; GFX1064: s_or_b64 [[OR:s\[[0-9:]+\]]],
; GFX1064: s_and_b64 exec, exec, [[OR]]
define amdgpu_gs void @test_kill_i1_terminator_i1(i32 %a, i32 %b, i32 %c, i32 %d) #0 {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = or i1 %c1, %c2
  call void @llvm.amdgcn.kill(i1 %x)
  ret void
}

; GCN-LABEL: {{^}}test_loop_vcc:
; GFX1032: v_cmp_lt_f32_e32 vcc_lo,
; GFX1064: v_cmp_lt_f32_e32 vcc,
; GCN: s_cbranch_vccnz
define amdgpu_ps <4 x float> @test_loop_vcc(<4 x float> %in) #0 {
entry:
  br label %loop

loop:
  %ctr.iv = phi float [ 0.0, %entry ], [ %ctr.next, %body ]
  %c.iv = phi <4 x float> [ %in, %entry ], [ %c.next, %body ]
  %cc = fcmp ogt float %ctr.iv, 7.0
  br i1 %cc, label %break, label %body

body:
  %c.iv0 = extractelement <4 x float> %c.iv, i32 0
  %c.next = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.iv0, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0)
  %ctr.next = fadd float %ctr.iv, 2.0
  br label %loop

break:
  ret <4 x float> %c.iv
}

; GCN-LABEL: {{^}}test_wwm1:
; GFX1032: s_or_saveexec_b32 [[SAVE:s[0-9]+]], -1
; GFX1032: s_mov_b32 exec_lo, [[SAVE]]
; GFX1064: s_or_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], -1
; GFX1064: s_mov_b64 exec, [[SAVE]]
define amdgpu_ps float @test_wwm1(i32 inreg %idx0, i32 inreg %idx1, float %src0, float %src1) {
main_body:
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  ret float %out.0
}

; GCN-LABEL: {{^}}test_wwm2:
; GFX1032: v_cmp_gt_u32_e32 vcc_lo, 32, v{{[0-9]+}}
; GFX1032: s_and_saveexec_b32 [[SAVE1:s[0-9]+]], vcc_lo
; GFX1032: s_or_saveexec_b32 [[SAVE2:s[0-9]+]], -1
; GFX1032: s_mov_b32 exec_lo, [[SAVE2]]
; GFX1032: s_or_b32 exec_lo, exec_lo, [[SAVE1]]
; GFX1064: v_cmp_gt_u32_e32 vcc, 32, v{{[0-9]+}}
; GFX1064: s_and_saveexec_b64 [[SAVE1:s\[[0-9:]+\]]], vcc{{$}}
; GFX1064: s_or_saveexec_b64 [[SAVE2:s\[[0-9:]+\]]], -1
; GFX1064: s_mov_b64 exec, [[SAVE2]]
; GFX1064: s_or_b64 exec, exec, [[SAVE1]]
define amdgpu_ps float @test_wwm2(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 32
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  %out.1 = fadd float %src, %out.0
  br label %endif

endif:
  %out.2 = phi float [ %out.1, %if ], [ 0.0, %main_body ]
  ret float %out.2
}

; GCN-LABEL: {{^}}test_wqm1:
; GFX1032: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GFX1032: s_wqm_b32 exec_lo, exec_lo
; GFX1032: s_and_b32 exec_lo, exec_lo, [[ORIG]]
; GFX1064: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec{{$}}
; GFX1064: s_wqm_b64 exec, exec{{$}}
; GFX1064: s_and_b64 exec, exec, [[ORIG]]
define amdgpu_ps <4 x float> @test_wqm1(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <2 x float> %pos) #0 {
main_body:
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  %inst28 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 1, i32 0, i32 %m0)
  %inst29 = tail call float @llvm.amdgcn.interp.p2(float %inst28, float %inst24, i32 1, i32 0, i32 %m0)
  %tex = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %inst26, float %inst29, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0)
  ret <4 x float> %tex
}

; GCN-LABEL: {{^}}test_wqm2:
; GFX1032: s_wqm_b32 exec_lo, exec_lo
; GFX1032: s_and_b32 exec_lo, exec_lo, s{{[0-9+]}}
; GFX1064: s_wqm_b64 exec, exec{{$}}
; GFX1064: s_and_b64 exec, exec, s[{{[0-9:]+}}]
define amdgpu_ps float @test_wqm2(i32 inreg %idx0, i32 inreg %idx1) #0 {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %out = fadd float %src0, %src1
  %out.0 = bitcast float %out to i32
  %out.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %out.0)
  %out.2 = bitcast i32 %out.1 to float
  ret float %out.2
}

; GCN-LABEL: {{^}}test_intr_fcmp_i64:
; GFX1032-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], 0{{$}}
; GFX1032-DAG: v_cmp_eq_f32_e64 s[[C_LO:[0-9]+]], {{s[0-9]+}}, |{{[vs][0-9]+}}|
; GFX1032-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]
; GFX1064:     v_cmp_eq_f32_e64 s{{\[}}[[C_LO:[0-9]+]]:[[C_HI:[0-9]+]]], {{s[0-9]+}}, |{{[vs][0-9]+}}|
; GFX1064-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]
; GFX1064-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[C_HI]]
; GCN:         store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[V_LO]]:[[V_HI]]],
define amdgpu_kernel void @test_intr_fcmp_i64(i64 addrspace(1)* %out, float %src, float %a) {
  %temp = call float @llvm.fabs.f32(float %a)
  %result = call i64 @llvm.amdgcn.fcmp.i64.f32(float %src, float %temp, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_intr_icmp_i64:
; GFX1032-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], 0{{$}}
; GFX1032-DAG: v_cmp_eq_u32_e64 [[C_LO:vcc_lo|s[0-9]+]], 0x64, {{s[0-9]+}}
; GFX1032-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[C_LO]]
; GFX1064:     v_cmp_eq_u32_e64 s{{\[}}[[C_LO:[0-9]+]]:[[C_HI:[0-9]+]]], 0x64, {{s[0-9]+}}
; GFX1064-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]
; GFX1064-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[C_HI]]
; GCN:         store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[V_LO]]:[[V_HI]]],
define amdgpu_kernel void @test_intr_icmp_i64(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64.i32(i32 %src, i32 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_intr_fcmp_i32:
; GFX1032-DAG: v_cmp_eq_f32_e64 s[[C_LO:[0-9]+]], {{s[0-9]+}}, |{{[vs][0-9]+}}|
; GFX1032-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]
; GFX1064:     v_cmp_eq_f32_e64 s{{\[}}[[C_LO:[0-9]+]]:[[C_HI:[0-9]+]]], {{s[0-9]+}}, |{{[vs][0-9]+}}|
; GFX1064-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]
; GCN:         store_dword v[{{[0-9:]+}}], v[[V_LO]],
define amdgpu_kernel void @test_intr_fcmp_i32(i32 addrspace(1)* %out, float %src, float %a) {
  %temp = call float @llvm.fabs.f32(float %a)
  %result = call i32 @llvm.amdgcn.fcmp.i32.f32(float %src, float %temp, i32 1)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_intr_icmp_i32:
; GFX1032-DAG: v_cmp_eq_u32_e64 s[[C_LO:[0-9]+]], 0x64, {{s[0-9]+}}
; GFX1032-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]{{$}}
; GFX1064:     v_cmp_eq_u32_e64 s{{\[}}[[C_LO:[0-9]+]]:{{[0-9]+}}], 0x64, {{s[0-9]+}}
; GFX1064-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[C_LO]]{{$}}
; GCN:         store_dword v[{{[0-9:]+}}], v[[V_LO]],
define amdgpu_kernel void @test_intr_icmp_i32(i32 addrspace(1)* %out, i32 %src) {
  %result = call i32 @llvm.amdgcn.icmp.i32.i32(i32 %src, i32 100, i32 32)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_wqm_vote:
; GFX1032: v_cmp_neq_f32_e32 vcc_lo, 0
; GFX1032: s_wqm_b32 [[WQM:s[0-9]+]], vcc_lo
; GFX1032: s_and_b32 exec_lo, exec_lo, [[WQM]]
; GFX1064: v_cmp_neq_f32_e32 vcc, 0
; GFX1064: s_wqm_b64 [[WQM:s\[[0-9:]+\]]], vcc{{$}}
; GFX1064: s_and_b64 exec, exec, [[WQM]]
define amdgpu_ps void @test_wqm_vote(float %a) {
  %c1 = fcmp une float %a, 0.0
  %c2 = call i1 @llvm.amdgcn.wqm.vote(i1 %c1)
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

; GCN-LABEL: {{^}}test_branch_true:
; GFX1032: s_and_b32 vcc_lo, exec_lo, -1
; GFX1064: s_and_b64 vcc, exec, -1
define amdgpu_kernel void @test_branch_true() #2 {
entry:
  br i1 true, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; GCN-LABEL: {{^}}test_ps_live:
; GFX1032: s_mov_b32 [[C:s[0-9]+]], exec_lo
; GFX1064: s_mov_b64 [[C:s\[[0-9:]+\]]], exec{{$}}
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[C]]
define amdgpu_ps float @test_ps_live() #0 {
  %live = call i1 @llvm.amdgcn.ps.live()
  %live.32 = zext i1 %live to i32
  %r = bitcast i32 %live.32 to float
  ret float %r
}

; GCN-LABEL: {{^}}test_vccnz_ifcvt_triangle64:
; GFX1032: v_cmp_neq_f64_e64 [[C:s[0-9]+]], s[{{[0-9:]+}}], 1.0
; GFX1032: s_and_b32 vcc_lo, exec_lo, [[C]]
; GFX1064: v_cmp_neq_f64_e64 [[C:s\[[0-9:]+\]]], s[{{[0-9:]+}}], 1.0
; GFX1064: s_and_b64 vcc, exec, [[C]]
define amdgpu_kernel void @test_vccnz_ifcvt_triangle64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
entry:
  %v = load double, double addrspace(1)* %in
  %cc = fcmp oeq double %v, 1.000000e+00
  br i1 %cc, label %if, label %endif

if:
  %u = fadd double %v, %v
  br label %endif

endif:
  %r = phi double [ %v, %entry ], [ %u, %if ]
  store double %r, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_vgprblocks_w32_attr:
; Test that the wave size can be overridden in function attributes and that the block size is correct as a result
; GFX10DEFWAVE: ; VGPRBlocks: 1
define amdgpu_gs float @test_vgprblocks_w32_attr(float %a, float %b, float %c, float %d, float %e,
                                        float %f, float %g, float %h, float %i, float %j, float %k, float %l) #3 {
main_body:
  %s = fadd float %a, %b
  %s.1 = fadd float %s, %c
  %s.2 = fadd float %s.1, %d
  %s.3 = fadd float %s.2, %e
  %s.4 = fadd float %s.3, %f
  %s.5 = fadd float %s.4, %g
  %s.6 = fadd float %s.5, %h
  %s.7 = fadd float %s.6, %i
  %s.8 = fadd float %s.7, %j
  %s.9 = fadd float %s.8, %k
  %s.10 = fadd float %s.9, %l
  ret float %s.10
}

; GCN-LABEL: {{^}}test_vgprblocks_w64_attr:
; Test that the wave size can be overridden in function attributes and that the block size is correct as a result
; GFX10DEFWAVE: ; VGPRBlocks: 2
define amdgpu_gs float @test_vgprblocks_w64_attr(float %a, float %b, float %c, float %d, float %e,
                                        float %f, float %g, float %h, float %i, float %j, float %k, float %l) #4 {
main_body:
  %s = fadd float %a, %b
  %s.1 = fadd float %s, %c
  %s.2 = fadd float %s.1, %d
  %s.3 = fadd float %s.2, %e
  %s.4 = fadd float %s.3, %f
  %s.5 = fadd float %s.4, %g
  %s.6 = fadd float %s.5, %h
  %s.7 = fadd float %s.6, %i
  %s.8 = fadd float %s.7, %j
  %s.9 = fadd float %s.8, %k
  %s.10 = fadd float %s.9, %l
  ret float %s.10
}

; GCN-LABEL: {{^}}icmp64:
; GFX1032: v_cmp_eq_u32_e32 vcc_lo, 0, v
; GFX1064: v_cmp_eq_u32_e32 vcc, 0, v
define amdgpu_kernel void @icmp64(i32 %n, i32 %s) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul4 = mul nsw i32 %s, %n
  %cmp = icmp slt i32 0, %mul4
  br label %if.end

if.end:                                           ; preds = %entry
  %rem = urem i32 %id, %s
  %icmp = tail call i64 @llvm.amdgcn.icmp.i64.i32(i32 %rem, i32 0, i32 32)
  %shr = lshr i64 %icmp, 1
  %notmask = shl nsw i64 -1, 0
  %and = and i64 %notmask, %shr
  %or = or i64 %and, -9223372036854775808
  %cttz = tail call i64 @llvm.cttz.i64(i64 %or, i1 true)
  %cast = trunc i64 %cttz to i32
  %cmp3 = icmp ugt i32 10, %cast
  %cmp6 = icmp ne i32 %rem, 0
  %brmerge = or i1 %cmp6, %cmp3
  br i1 %brmerge, label %if.end2, label %if.then

if.then:                                          ; preds = %if.end
  unreachable

if.end2:                                          ; preds = %if.end
  ret void
}

; GCN-LABEL: {{^}}fcmp64:
; GFX1032: v_cmp_eq_f32_e32 vcc_lo, 0, v
; GFX1064: v_cmp_eq_f32_e32 vcc, 0, v
define amdgpu_kernel void @fcmp64(float %n, float %s) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %id.f = uitofp i32 %id to float
  %mul4 = fmul float %s, %n
  %cmp = fcmp ult float 0.0, %mul4
  br label %if.end

if.end:                                           ; preds = %entry
  %rem.f = frem float %id.f, %s
  %fcmp = tail call i64 @llvm.amdgcn.fcmp.i64.f32(float %rem.f, float 0.0, i32 1)
  %shr = lshr i64 %fcmp, 1
  %notmask = shl nsw i64 -1, 0
  %and = and i64 %notmask, %shr
  %or = or i64 %and, -9223372036854775808
  %cttz = tail call i64 @llvm.cttz.i64(i64 %or, i1 true)
  %cast = trunc i64 %cttz to i32
  %cmp3 = icmp ugt i32 10, %cast
  %cmp6 = fcmp one float %rem.f, 0.0
  %brmerge = or i1 %cmp6, %cmp3
  br i1 %brmerge, label %if.end2, label %if.then

if.then:                                          ; preds = %if.end
  unreachable

if.end2:                                          ; preds = %if.end
  ret void
}

; GCN-LABEL: {{^}}icmp32:
; GFX1032: v_cmp_eq_u32_e32 vcc_lo, 0, v
; GFX1064: v_cmp_eq_u32_e32 vcc, 0, v
define amdgpu_kernel void @icmp32(i32 %n, i32 %s) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul4 = mul nsw i32 %s, %n
  %cmp = icmp slt i32 0, %mul4
  br label %if.end

if.end:                                           ; preds = %entry
  %rem = urem i32 %id, %s
  %icmp = tail call i32 @llvm.amdgcn.icmp.i32.i32(i32 %rem, i32 0, i32 32)
  %shr = lshr i32 %icmp, 1
  %notmask = shl nsw i32 -1, 0
  %and = and i32 %notmask, %shr
  %or = or i32 %and, 2147483648
  %cttz = tail call i32 @llvm.cttz.i32(i32 %or, i1 true)
  %cmp3 = icmp ugt i32 10, %cttz
  %cmp6 = icmp ne i32 %rem, 0
  %brmerge = or i1 %cmp6, %cmp3
  br i1 %brmerge, label %if.end2, label %if.then

if.then:                                          ; preds = %if.end
  unreachable

if.end2:                                          ; preds = %if.end
  ret void
}

; GCN-LABEL: {{^}}fcmp32:
; GFX1032: v_cmp_eq_f32_e32 vcc_lo, 0, v
; GFX1064: v_cmp_eq_f32_e32 vcc, 0, v
define amdgpu_kernel void @fcmp32(float %n, float %s) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %id.f = uitofp i32 %id to float
  %mul4 = fmul float %s, %n
  %cmp = fcmp ult float 0.0, %mul4
  br label %if.end

if.end:                                           ; preds = %entry
  %rem.f = frem float %id.f, %s
  %fcmp = tail call i32 @llvm.amdgcn.fcmp.i32.f32(float %rem.f, float 0.0, i32 1)
  %shr = lshr i32 %fcmp, 1
  %notmask = shl nsw i32 -1, 0
  %and = and i32 %notmask, %shr
  %or = or i32 %and, 2147483648
  %cttz = tail call i32 @llvm.cttz.i32(i32 %or, i1 true)
  %cmp3 = icmp ugt i32 10, %cttz
  %cmp6 = fcmp one float %rem.f, 0.0
  %brmerge = or i1 %cmp6, %cmp3
  br i1 %brmerge, label %if.end2, label %if.then

if.then:                                          ; preds = %if.end
  unreachable

if.end2:                                          ; preds = %if.end
  ret void
}

declare void @external_void_func_void() #1

; Test save/restore of VGPR needed for SGPR spilling.

; GCN-LABEL: {{^}}callee_no_stack_with_call:
; GCN: s_waitcnt
; GCN-NEXT: s_waitcnt_vscnt

; GFX1064-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GFX1032-NEXT: s_or_saveexec_b32 [[COPY_EXEC0:s[0-9]]], -1{{$}}
; GCN-NEXT: buffer_store_dword v32, off, s[0:3], s32 ; 4-byte Folded Spill
; GCN-NEXT: v_nop
; GFX1064-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GFX1032-NEXT: s_mov_b32 exec_lo, [[COPY_EXEC0]]

; GCN-NEXT: v_writelane_b32 v32, s34, 2
; GCN: s_mov_b32 s34, s32
; GFX1064: s_add_u32 s32, s32, 0x400
; GFX1032: s_add_u32 s32, s32, 0x200


; GCN-DAG: v_writelane_b32 v32, s30, 0
; GCN-DAG: v_writelane_b32 v32, s31, 1
; GCN: s_swappc_b64
; GCN-DAG: v_readlane_b32 s4, v32, 0
; GCN-DAG: v_readlane_b32 s5, v32, 1


; GFX1064: s_sub_u32 s32, s32, 0x400
; GFX1032: s_sub_u32 s32, s32, 0x200
; GCN: v_readlane_b32 s34, v32, 2
; GFX1064: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GFX1032: s_or_saveexec_b32 [[COPY_EXEC1:s[0-9]]], -1{{$}}
; GCN-NEXT: buffer_load_dword v32, off, s[0:3], s32 ; 4-byte Folded Reload
; GCN-NEXT: v_nop
; GFX1064-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GFX1032-NEXT: s_mov_b32 exec_lo, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_with_call() #1 {
  call void @external_void_func_void()
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.fabs.f32(float)
declare { float, i1 } @llvm.amdgcn.div.scale.f32(float, float, i1)
declare { double, i1 } @llvm.amdgcn.div.scale.f64(double, double, i1)
declare float @llvm.amdgcn.div.fmas.f32(float, float, float, i1)
declare double @llvm.amdgcn.div.fmas.f64(double, double, double, i1)
declare i1 @llvm.amdgcn.class.f32(float, i32)
declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32)
declare i64 @llvm.amdgcn.set.inactive.i64(i64, i64)
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare float @llvm.amdgcn.wwm.f32(float)
declare i32 @llvm.amdgcn.wqm.i32(i32)
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32)
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32)
declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
declare i64 @llvm.amdgcn.fcmp.i64.f32(float, float, i32)
declare i64 @llvm.amdgcn.icmp.i64.i32(i32, i32, i32)
declare i32 @llvm.amdgcn.fcmp.i32.f32(float, float, i32)
declare i32 @llvm.amdgcn.icmp.i32.i32(i32, i32, i32)
declare void @llvm.amdgcn.kill(i1)
declare i1 @llvm.amdgcn.wqm.vote(i1)
declare i1 @llvm.amdgcn.ps.live()
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone optnone noinline }
attributes #3 = { "target-features"="+wavefrontsize32" }
attributes #4 = { "target-features"="+wavefrontsize64" }
