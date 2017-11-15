; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Test that doing a shift of a pointer with a constant add will be
; folded into the constant offset addressing mode even if the add has
; multiple uses. This is relevant to accessing 2 separate, adjacent
; LDS globals.


declare i32 @llvm.amdgcn.workitem.id.x() #1

@lds0 = addrspace(3) global [512 x float] undef, align 4
@lds1 = addrspace(3) global [512 x float] undef, align 4


; Make sure the (add tid, 2) << 2 gets folded into the ds's offset as (tid << 2) + 8

; GCN-LABEL: {{^}}load_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_read_b32 {{v[0-9]+}}, [[PTR]] offset:8
; GCN: s_endpgm
define amdgpu_kernel void @load_shl_base_lds_0(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  store float %val0, float addrspace(1)* %out
  ret void
}

; Make sure once the first use is folded into the addressing mode, the
; remaining add use goes through the normal shl + add constant fold.

; GCN-LABEL: {{^}}load_shl_base_lds_1:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_read_b32 [[RESULT:v[0-9]+]], [[PTR]] offset:8
; GCN: v_add_i32_e32 [[ADDUSE:v[0-9]+]], vcc, 8, v{{[0-9]+}}
; GCN-DAG: buffer_store_dword [[RESULT]]
; GCN-DAG: buffer_store_dword [[ADDUSE]]
; GCN: s_endpgm
define amdgpu_kernel void @load_shl_base_lds_1(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %shl_add_use = shl i32 %idx.0, 2
  store i32 %shl_add_use, i32 addrspace(1)* %add_use, align 4
  store float %val0, float addrspace(1)* %out
  ret void
}

@maxlds = addrspace(3) global [65536 x i8] undef, align 4

; GCN-LABEL: {{^}}load_shl_base_lds_max_offset
; GCN: ds_read_u8 v{{[0-9]+}}, v{{[0-9]+}} offset:65535
; GCN: s_endpgm
define amdgpu_kernel void @load_shl_base_lds_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %lds, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 65535
  %arrayidx0 = getelementptr inbounds [65536 x i8], [65536 x i8] addrspace(3)* @maxlds, i32 0, i32 %idx.0
  %val0 = load i8, i8 addrspace(3)* %arrayidx0
  store i32 %idx.0, i32 addrspace(1)* %add_use
  store i8 %val0, i8 addrspace(1)* %out
  ret void
}

; The two globals are placed adjacent in memory, so the same base
; pointer can be used with an offset into the second one.

; GCN-LABEL: {{^}}load_shl_base_lds_2:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: s_mov_b32 m0, -1
; GCN-NEXT: ds_read2st64_b32 {{v\[[0-9]+:[0-9]+\]}}, [[PTR]] offset0:1 offset1:9
; GCN: s_endpgm
define amdgpu_kernel void @load_shl_base_lds_2(float addrspace(1)* %out) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 64
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds1, i32 0, i32 %idx.0
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  store float %sum, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}store_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_write_b32 [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @store_shl_base_lds_0(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  store float 1.0, float addrspace(3)* %arrayidx0, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}


; --------------------------------------------------------------------------------
; Atomics.

@lds2 = addrspace(3) global [512 x i32] undef, align 4

; define amdgpu_kernel void @atomic_load_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
;   %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
;   %idx.0 = add nsw i32 %tid.x, 2
;   %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
;   %val = load atomic i32, i32 addrspace(3)* %arrayidx0 seq_cst, align 4
;   store i32 %val, i32 addrspace(1)* %out, align 4
;   store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
;   ret void
; }


; GCN-LABEL: {{^}}atomic_cmpxchg_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_cmpst_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}}, {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_cmpxchg_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use, i32 %swap) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %pair = cmpxchg i32 addrspace(3)* %arrayidx0, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_swap_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_wrxchg_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_swap_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw xchg i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_add_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_add_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_add_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw add i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_sub_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_sub_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw sub i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_and_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_and_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_and_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw and i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_or_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_or_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_or_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw or i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_xor_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_xor_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw xor i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; define amdgpu_kernel void @atomic_nand_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
;   %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
;   %idx.0 = add nsw i32 %tid.x, 2
;   %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
;   %val = atomicrmw nand i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
;   store i32 %val, i32 addrspace(1)* %out, align 4
;   store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
;   ret void
; }

; GCN-LABEL: {{^}}atomic_min_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_min_rtn_i32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_min_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw min i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_max_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_max_rtn_i32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_max_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw max i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_min_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_umin_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw umin i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_shl_base_lds_0:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_max_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; GCN: s_endpgm
define amdgpu_kernel void @atomic_umax_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw umax i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_lds:
; GCN: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 3, v0
; GCN: ds_write_b32 [[SCALE0]], v{{[0-9]+}} offset:32

; GCN: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 4, v0
; GCN: ds_write_b32 [[SCALE1]], v{{[0-9]+}} offset:64
define void @shl_add_ptr_combine_2use_lds(i32 %idx) #0 {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to i32 addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to i32 addrspace(3)*
  store volatile i32 9, i32 addrspace(3)* %ptr0
  store volatile i32 10, i32 addrspace(3)* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_max_lds_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 3, v0
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 4, v0
; GCN-DAG: ds_write_b32 [[SCALE0]], v{{[0-9]+}} offset:65528
; GCN-DAG: v_add_i32_e32 [[ADD1:v[0-9]+]], vcc, 0x1fff0, [[SCALE1]]
; GCN: ds_write_b32 [[ADD1]], v{{[0-9]+$}}
define void @shl_add_ptr_combine_2use_max_lds_offset(i32 %idx) #0 {
  %idx.add = add nuw i32 %idx, 8191
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to i32 addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to i32 addrspace(3)*
  store volatile i32 9, i32 addrspace(3)* %ptr0
  store volatile i32 10, i32 addrspace(3)* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_both_max_lds_offset:
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, 0x1000, v0
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 4, [[ADD]]
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 5, [[ADD]]
; GCN-DAG: ds_write_b32 [[SCALE0]], v{{[0-9]+$}}
; GCN: ds_write_b32 [[SCALE1]], v{{[0-9]+$}}
define void @shl_add_ptr_combine_2use_both_max_lds_offset(i32 %idx) #0 {
  %idx.add = add nuw i32 %idx, 4096
  %shl0 = shl i32 %idx.add, 4
  %shl1 = shl i32 %idx.add, 5
  %ptr0 = inttoptr i32 %shl0 to i32 addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to i32 addrspace(3)*
  store volatile i32 9, i32 addrspace(3)* %ptr0
  store volatile i32 10, i32 addrspace(3)* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_private:
; GCN: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 2, v0
; GCN: buffer_store_dword v{{[0-9]+}}, [[SCALE0]], s[0:3], s4 offen offset:16

; GCN: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 3, v0
; GCN: buffer_store_dword v{{[0-9]+}}, [[SCALE1]], s[0:3], s4 offen offset:32
define void @shl_add_ptr_combine_2use_private(i16 zeroext %idx.arg) #0 {
  %idx = zext i16 %idx.arg to i32
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 2
  %shl1 = shl i32 %idx.add, 3
  %ptr0 = inttoptr i32 %shl0 to i32*
  %ptr1 = inttoptr i32 %shl1 to i32*
  store volatile i32 9, i32* %ptr0
  store volatile i32 10, i32* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_max_private_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 3, v0
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 4, v0
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, [[SCALE0]], s[0:3], s4 offen offset:4088
; GCN-DAG: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, 0x1ff0, [[SCALE1]]
; GCN: buffer_store_dword v{{[0-9]+}}, [[ADD]], s[0:3], s4 offen{{$}}
define void @shl_add_ptr_combine_2use_max_private_offset(i16 zeroext %idx.arg) #0 {
  %idx = zext i16 %idx.arg to i32
  %idx.add = add nuw i32 %idx, 511
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to i32*
  %ptr1 = inttoptr i32 %shl1 to i32*
  store volatile i32 9, i32* %ptr0
  store volatile i32 10, i32* %ptr1
  ret void
}
; GCN-LABEL: {{^}}shl_add_ptr_combine_2use_both_max_private_offset:
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, 0x100, v0
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 4, [[ADD]]
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 5, [[ADD]]
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, [[SCALE0]], s[0:3], s4 offen{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, [[SCALE1]], s[0:3], s4 offen{{$}}
define void @shl_add_ptr_combine_2use_both_max_private_offset(i16 zeroext %idx.arg) #0 {
  %idx = zext i16 %idx.arg to i32
  %idx.add = add nuw i32 %idx, 256
  %shl0 = shl i32 %idx.add, 4
  %shl1 = shl i32 %idx.add, 5
  %ptr0 = inttoptr i32 %shl0 to i32*
  %ptr1 = inttoptr i32 %shl1 to i32*
  store volatile i32 9, i32* %ptr0
  store volatile i32 10, i32* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_or_ptr_combine_2use_lds:
; GCN: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 3, v0
; GCN: ds_write_b32 [[SCALE0]], v{{[0-9]+}} offset:32

; GCN: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 4, v0
; GCN: ds_write_b32 [[SCALE1]], v{{[0-9]+}} offset:64
define void @shl_or_ptr_combine_2use_lds(i32 %idx) #0 {
  %idx.add = or i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to i32 addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to i32 addrspace(3)*
  store volatile i32 9, i32 addrspace(3)* %ptr0
  store volatile i32 10, i32 addrspace(3)* %ptr1
  ret void
}

; GCN-LABEL: {{^}}shl_or_ptr_combine_2use_max_lds_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE0:v[0-9]+]], 3, v0
; GCN-DAG: v_lshlrev_b32_e32 [[SCALE1:v[0-9]+]], 4, v0
; GCN-DAG: ds_write_b32 [[SCALE0]], v{{[0-9]+}} offset:65528
; GCN-DAG: v_or_b32_e32 [[ADD1:v[0-9]+]], 0x1fff0, [[SCALE1]]
; GCN: ds_write_b32 [[ADD1]], v{{[0-9]+$}}
define void @shl_or_ptr_combine_2use_max_lds_offset(i32 %idx) #0 {
  %idx.add = or i32 %idx, 8191
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to i32 addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to i32 addrspace(3)*
  store volatile i32 9, i32 addrspace(3)* %ptr0
  store volatile i32 10, i32 addrspace(3)* %ptr1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
