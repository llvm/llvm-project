; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-code-object-v3,-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=CI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3,-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=GFX9 %s

; HSA-LABEL: {{^}}use_group_to_flat_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; CI: enable_sgpr_queue_ptr = 1
; GFX9: enable_sgpr_queue_ptr = 0

; CI-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10{{$}}
; CI-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[APERTURE]]
; CI-DAG: v_cmp_ne_u32_e64 vcc, [[PTR]], -1
; CI-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]], vcc
; CI-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; CI-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]

; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; GFX9-DAG: s_load_dword [[PTR:s[0-9]+]], s[4:5], 0x0{{$}}
; GFX9-DAG: s_getreg_b32 [[SSRC_SHARED:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; GFX9-DAG: s_lshl_b32 [[SSRC_SHARED_BASE:s[0-9]+]], [[SSRC_SHARED]], 16
; GFX9-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[SSRC_SHARED_BASE]]

; GFX9-XXX: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], src_shared_base
; GFX9: v_cmp_ne_u32_e64 vcc, [[PTR]], -1
; GFX9: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]], vcc
; GFX9-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GFX9-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]

; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, [[K]]

; At most 2 digits. Make sure src_shared_base is not counted as a high
; number SGPR.

; CI: NumSgprs: {{[0-9][0-9]+}}
; GFX9: NumSgprs: {{[0-9]+}}
define amdgpu_kernel void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32*
  store volatile i32 7, i32* %stof
  ret void
}

; HSA-LABEL: {{^}}use_private_to_flat_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; CI: enable_sgpr_queue_ptr = 1
; GFX9: enable_sgpr_queue_ptr = 0

; CI-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x11{{$}}
; CI-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[APERTURE]]

; CI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; CI-DAG: v_cmp_ne_u32_e64 vcc, [[PTR]], 0
; CI-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]], vcc
; CI-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; CI-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]

; GFX9-DAG: s_load_dword [[PTR:s[0-9]+]], s[4:5], 0x0{{$}}
; GFX9-DAG: s_getreg_b32 [[SSRC_PRIVATE:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 0, 16)
; GFX9-DAG: s_lshl_b32 [[SSRC_PRIVATE_BASE:s[0-9]+]], [[SSRC_PRIVATE]], 16
; GFX9-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[SSRC_PRIVATE_BASE]]

; GFX9-XXX: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], src_private_base

; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; GFX9: v_cmp_ne_u32_e64 vcc, [[PTR]], 0
; GFX9: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]], vcc
; GFX9: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GFX9-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]

; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, [[K]]

; CI: NumSgprs: {{[0-9][0-9]+}}
; GFX9: NumSgprs: {{[0-9]+}}
define amdgpu_kernel void @use_private_to_flat_addrspacecast(i32 addrspace(5)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(5)* %ptr to i32*
  store volatile i32 7, i32* %stof
  ret void
}

; no-op
; HSA-LABEL: {{^}}use_global_to_flat_addrspacecast:
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; HSA: flat_store_dword v{{\[}}[[VPTRLO]]:[[VPTRHI]]{{\]}}, [[K]]
define amdgpu_kernel void @use_global_to_flat_addrspacecast(i32 addrspace(1)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(1)* %ptr to i32*
  store volatile i32 7, i32* %stof
  ret void
}

; no-op
; HSA-LABEl: {{^}}use_constant_to_flat_addrspacecast:
; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA: flat_load_dword v{{[0-9]+}}, v{{\[}}[[VPTRLO]]:[[VPTRHI]]{{\]}}
define amdgpu_kernel void @use_constant_to_flat_addrspacecast(i32 addrspace(4)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(4)* %ptr to i32*
  %ld = load volatile i32, i32* %stof
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_group_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]{{\]}}
; HSA-DAG: v_cmp_ne_u64_e64 vcc, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], s[[PTR_LO]]
; HSA-DAG: v_cndmask_b32_e32 [[CASTPTR:v[0-9]+]], -1, v[[VPTR_LO]]
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 0{{$}}
; HSA: ds_write_b32 [[CASTPTR]], v[[K]]
define amdgpu_kernel void @use_flat_to_group_addrspacecast(i32* %ptr) #0 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(3)*
  store volatile i32 0, i32 addrspace(3)* %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_private_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]{{\]}}
; HSA-DAG: v_cmp_ne_u64_e64 vcc, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], s[[PTR_LO]]
; HSA-DAG: v_cndmask_b32_e32 [[CASTPTR:v[0-9]+]], 0, v[[VPTR_LO]]
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 0{{$}}
; HSA: buffer_store_dword v[[K]], [[CASTPTR]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen{{$}}
define amdgpu_kernel void @use_flat_to_private_addrspacecast(i32* %ptr) #0 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(5)*
  store volatile i32 0, i32 addrspace(5)* %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_global_addrspacecast:
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}, s[4:5], 0x0
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0
; HSA: {{flat|global}}_store_dword v{{\[}}[[VPTRLO]]:[[VPTRHI]]{{\]}}, [[K]]
define amdgpu_kernel void @use_flat_to_global_addrspacecast(i32* %ptr) #0 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(1)*
  store volatile i32 0, i32 addrspace(1)* %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_constant_addrspacecast:
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}, s[4:5], 0x0
; HSA: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTRLO]]:[[PTRHI]]{{\]}}, 0x0
define amdgpu_kernel void @use_flat_to_constant_addrspacecast(i32* %ptr) #0 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(4)*
  load volatile i32, i32 addrspace(4)* %ftos
  ret void
}

; HSA-LABEL: {{^}}cast_0_group_to_flat_addrspacecast:
; CI: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10
; CI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE]]
; GFX9-DAG: s_getreg_b32 [[SSRC_SHARED:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; GFX9-DAG: s_lshl_b32 [[SSRC_SHARED_BASE:s[0-9]+]], [[SSRC_SHARED]], 16
; GFX9-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[SSRC_SHARED_BASE]]

; GFX9-XXX: v_mov_b32_e32 v[[HI:[0-9]+]], src_shared_base

; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: {{flat|global}}_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define amdgpu_kernel void @cast_0_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(3)* null to i32*
  store volatile i32 7, i32* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define amdgpu_kernel void @cast_0_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast i32* null to i32 addrspace(3)*
  store volatile i32 7, i32 addrspace(3)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_group_to_flat_addrspacecast:
; HSA: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA: {{flat|global}}_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define amdgpu_kernel void @cast_neg1_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(3)* inttoptr (i32 -1 to i32 addrspace(3)*) to i32*
  store volatile i32 7, i32* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define amdgpu_kernel void @cast_neg1_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast i32* inttoptr (i64 -1 to i32*) to i32 addrspace(3)*
  store volatile i32 7, i32 addrspace(3)* %cast
  ret void
}

; FIXME: Shouldn't need to enable queue ptr
; HSA-LABEL: {{^}}cast_0_private_to_flat_addrspacecast:
; CI: enable_sgpr_queue_ptr = 1
; GFX9: enable_sgpr_queue_ptr = 0

; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA: {{flat|global}}_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define amdgpu_kernel void @cast_0_private_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(5)* null to i32*
  store volatile i32 7, i32* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_private_addrspacecast:
; HSA: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: buffer_store_dword [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0
define amdgpu_kernel void @cast_0_flat_to_private_addrspacecast() #0 {
  %cast = addrspacecast i32* null to i32 addrspace(5)*
  store volatile i32 7, i32 addrspace(5)* %cast
  ret void
}

; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.

; HSA-LABEL: {{^}}branch_use_flat_i32:
; HSA: {{flat|global}}_store_dword {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}
; HSA: s_endpgm
define amdgpu_kernel void @branch_use_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* %gptr, i32 addrspace(3)* %lptr, i32 %x, i32 %c) #0 {
entry:
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %local, label %global

local:
  %flat_local = addrspacecast i32 addrspace(3)* %lptr to i32*
  br label %end

global:
  %flat_global = addrspacecast i32 addrspace(1)* %gptr to i32*
  br label %end

end:
  %fptr = phi i32* [ %flat_local, %local ], [ %flat_global, %global ]
  store volatile i32 %x, i32* %fptr, align 4
;  %val = load i32, i32* %fptr, align 4
;  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; Check for prologue initializing special SGPRs pointing to scratch.
; HSA-LABEL: {{^}}store_flat_scratch:
; CI-DAG: s_mov_b32 flat_scratch_lo, s9
; CI-DAG: s_add_u32 [[ADD:s[0-9]+]], s8, s11
; CI: s_lshr_b32 flat_scratch_hi, [[ADD]], 8

; GFX9: s_add_u32 flat_scratch_lo, s6, s9
; GFX9: s_addc_u32 flat_scratch_hi, s7, 0

; HSA: {{flat|global}}_store_dword
; HSA: s_barrier
; HSA: {{flat|global}}_load_dword
define amdgpu_kernel void @store_flat_scratch(i32 addrspace(1)* noalias %out, i32) #0 {
  %alloca = alloca i32, i32 9, align 4, addrspace(5)
  %x = call i32 @llvm.amdgcn.workitem.id.x() #2
  %pptr = getelementptr i32, i32 addrspace(5)* %alloca, i32 %x
  %fptr = addrspacecast i32 addrspace(5)* %pptr to i32*
  store volatile i32 %x, i32* %fptr
  ; Dummy call
  call void @llvm.amdgcn.s.barrier() #1
  %reload = load volatile i32, i32* %fptr, align 4
  store volatile i32 %reload, i32 addrspace(1)* %out, align 4
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind readnone }
