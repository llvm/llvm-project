; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt -enable-misched < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=+load-store-opt -enable-misched < %s | FileCheck -check-prefix=SI %s

; Test that doing a shift of a pointer with a constant add will be
; folded into the constant offset addressing mode even if the add has
; multiple uses. This is relevant to accessing 2 separate, adjacent
; LDS globals.


declare i32 @llvm.amdgcn.workitem.id.x() #1

@lds0 = addrspace(3) global [512 x float] undef, align 4
@lds1 = addrspace(3) global [512 x float] undef, align 4


; Make sure the (add tid, 2) << 2 gets folded into the ds's offset as (tid << 2) + 8

; SI-LABEL: {{^}}load_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_read_b32 {{v[0-9]+}}, [[PTR]] offset:8
; SI: s_endpgm
define void @load_shl_base_lds_0(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
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

; SI-LABEL: {{^}}load_shl_base_lds_1:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_read_b32 [[RESULT:v[0-9]+]], [[PTR]] offset:8
; SI: v_add_i32_e32 [[ADDUSE:v[0-9]+]], vcc, 8, v{{[0-9]+}}
; SI-DAG: buffer_store_dword [[RESULT]]
; SI-DAG: buffer_store_dword [[ADDUSE]]
; SI: s_endpgm
define void @load_shl_base_lds_1(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
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

; SI-LABEL: {{^}}load_shl_base_lds_max_offset
; SI: ds_read_u8 v{{[0-9]+}}, v{{[0-9]+}} offset:65535
; SI: s_endpgm
define void @load_shl_base_lds_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %lds, i32 addrspace(1)* %add_use) #0 {
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

; SI-LABEL: {{^}}load_shl_base_lds_2:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: s_mov_b32 m0, -1
; SI-NEXT: ds_read2st64_b32 {{v\[[0-9]+:[0-9]+\]}}, [[PTR]] offset0:1 offset1:9
; SI: s_endpgm
define void @load_shl_base_lds_2(float addrspace(1)* %out) #0 {
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

; SI-LABEL: {{^}}store_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_write_b32 [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @store_shl_base_lds_0(float addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
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

; define void @atomic_load_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
;   %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
;   %idx.0 = add nsw i32 %tid.x, 2
;   %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
;   %val = load atomic i32, i32 addrspace(3)* %arrayidx0 seq_cst, align 4
;   store i32 %val, i32 addrspace(1)* %out, align 4
;   store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
;   ret void
; }


; SI-LABEL: {{^}}atomic_cmpxchg_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_cmpst_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}}, {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_cmpxchg_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use, i32 %swap) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %pair = cmpxchg i32 addrspace(3)* %arrayidx0, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_swap_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_wrxchg_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_swap_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw xchg i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_add_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_add_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_add_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw add i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_sub_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_sub_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_sub_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw sub i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_and_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_and_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_and_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw and i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_or_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_or_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_or_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw or i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_xor_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_xor_rtn_b32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_xor_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw xor i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; define void @atomic_nand_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
;   %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
;   %idx.0 = add nsw i32 %tid.x, 2
;   %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
;   %val = atomicrmw nand i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
;   store i32 %val, i32 addrspace(1)* %out, align 4
;   store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
;   ret void
; }

; SI-LABEL: {{^}}atomic_min_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_min_rtn_i32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_min_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw min i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_max_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_max_rtn_i32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_max_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw max i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_umin_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_min_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_umin_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw umin i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

; SI-LABEL: {{^}}atomic_umax_shl_base_lds_0:
; SI: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; SI: ds_max_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
; SI: s_endpgm
define void @atomic_umax_shl_base_lds_0(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds2, i32 0, i32 %idx.0
  %val = atomicrmw umax i32 addrspace(3)* %arrayidx0, i32 3 seq_cst
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i32 %idx.0, i32 addrspace(1)* %add_use, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
