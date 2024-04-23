; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=CI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=GFX9 %s

; HSA-LABEL: {{^}}use_group_to_flat_addrspacecast:

; CI-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10{{$}}
; CI-DAG: s_cmp_lg_u32 [[PTR]], -1
; CI-DAG: s_cselect_b32 s[[HI:[0-9]+]], [[APERTURE]], 0
; CI-DAG: s_cselect_b32 s[[LO:[0-9]+]], [[PTR]], 0

; GFX9-DAG: s_mov_b64 s[{{[0-9]+}}:[[HIBASE:[0-9]+]]], src_shared_base

; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; GFX9-DAG: s_load_dword [[PTR:s[0-9]+]], s[4:5], 0x0{{$}}

; GFX9: s_cmp_lg_u32 [[PTR]], -1
; GFX9-DAG: s_cselect_b32 s[[LO:[0-9]+]], s[[HIBASE]], 0
; GFX9-DAG: s_cselect_b32 s[[HI:[0-9]+]], [[PTR]], 0

; HSA: flat_store_dword v[[[LO]]:[[HI]]], [[K]]

; HSA:  .amdhsa_user_sgpr_private_segment_buffer 1
; HSA:  .amdhsa_user_sgpr_dispatch_ptr 0
; CI:   .amdhsa_user_sgpr_queue_ptr 1
; GFX9: .amdhsa_user_sgpr_queue_ptr 0

; At most 2 digits. Make sure src_shared_base is not counted as a high
; number SGPR.

; HSA: NumSgprs: {{[0-9]+}}
define amdgpu_kernel void @use_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 7, ptr %stof
  ret void
}

; Test handling inside a non-kernel
; HSA-LABEL: {{^}}use_group_to_flat_addrspacecast_func:
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[6:7], 0x10{{$}}
; CI-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[APERTURE]]
; CI-DAG: v_cmp_ne_u32_e32 vcc, -1, v0
; CI-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]], vcc
; CI-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, v0

; GFX9-DAG: s_mov_b64 s[{{[0-9]+}}:[[HIBASE:[0-9]+]]], src_shared_base

; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7

; GFX9-DAG: v_mov_b32_e32 v[[VREG_HIBASE:[0-9]+]], s[[HIBASE]]
; GFX9-DAG: v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, v0, vcc
; GFX9-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, v[[VREG_HIBASE]], vcc

; HSA: flat_store_dword v[[[LO]]:[[HI]]], [[K]]
define void @use_group_to_flat_addrspacecast_func(ptr addrspace(3) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 7, ptr %stof
  ret void
}

; HSA-LABEL: {{^}}use_private_to_flat_addrspacecast:

; CI-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x11{{$}}

; CI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; CI-DAG: s_cmp_lg_u32 [[PTR]], -1
; CI-DAG: s_cselect_b32 s[[HI:[0-9]+]], [[APERTURE]], 0
; CI-DAG: s_cselect_b32 s[[LO:[0-9]+]], [[PTR]], 0

; GFX9-DAG: s_load_dword [[PTR:s[0-9]+]], s[4:5], 0x0{{$}}
; GFX9-DAG: s_mov_b64 s[{{[0-9]+}}:[[HIBASE:[0-9]+]]], src_private_base

; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; GFX9: s_cmp_lg_u32 [[PTR]], -1
; GFX9: s_cselect_b32 s[[LO:[0-9]+]], s[[HIBASE]], 0
; GFX9: s_cselect_b32 s[[HI:[0-9]+]], [[PTR]], 0

; HSA: flat_store_dword v[[[LO]]:[[HI]]], [[K]]

; HSA:  .amdhsa_user_sgpr_private_segment_buffer 1
; HSA:  .amdhsa_user_sgpr_dispatch_ptr 0
; CI:   .amdhsa_user_sgpr_queue_ptr 1
; GFX9: .amdhsa_user_sgpr_queue_ptr 0

; HSA: NumSgprs: {{[0-9]+}}
define amdgpu_kernel void @use_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 7, ptr %stof
  ret void
}

; no-op
; HSA-LABEL: {{^}}use_global_to_flat_addrspacecast:

; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7
; HSA: flat_store_dword v[[[VPTRLO]]:[[VPTRHI]]], [[K]]

; HSA:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @use_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 7, ptr %stof
  ret void
}

; no-op
; HSA-LABEl: {{^}}use_constant_to_flat_addrspacecast:
; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA: flat_load_dword v{{[0-9]+}}, v[[[VPTRLO]]:[[VPTRHI]]]
define amdgpu_kernel void @use_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  %ld = load volatile i32, ptr %stof
  ret void
}

; HSA-LABEl: {{^}}use_constant_to_global_addrspacecast:
; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]]
; CI-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; CI-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; CI: {{flat|global}}_load_dword v{{[0-9]+}}, v[[[VPTRLO]]:[[VPTRHI]]]

; GFX9: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GFX9: global_load_dword v{{[0-9]+}}, [[ZERO:v[0-9]+]], s[[[PTRLO]]:[[PTRHI]]]
define amdgpu_kernel void @use_constant_to_global_addrspacecast(ptr addrspace(4) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr addrspace(1)
  %ld = load volatile i32, ptr addrspace(1) %stof
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_group_addrspacecast:

; HSA: s_load_dwordx2 s[[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]]
; CI-DAG: v_cmp_ne_u64_e64 s[[[CMP_LO:[0-9]+]]:[[CMP_HI:[0-9]+]]], s[[[PTR_LO]]:[[PTR_HI]]], 0{{$}}
; CI-DAG: s_and_b64 s{{[[0-9]+:[0-9]+]}}, s[[[CMP_LO]]:[[CMP_HI]]], exec
; CI-DAG: s_cselect_b32 [[CASTPTR:s[0-9]+]], s[[PTR_LO]], -1
; CI-DAG: v_mov_b32_e32 [[VCASTPTR:v[0-9]+]], [[CASTPTR]]
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 0{{$}}
; GFX9-DAG: s_cmp_lg_u64 s[[[CMP_LO:[0-9]+]]:[[CMP_HI:[0-9]+]]], 0
; GFX9-DAG: s_cselect_b32 s[[PTR_LO]], s[[PTR_LO]], -1
; GFX9-DAG: v_mov_b32_e32 [[CASTPTR:v[0-9]+]], s[[PTR_LO]]
; CI-DAG: ds_write_b32 [[VCASTPTR]], v[[K]]
; GFX9-DAG: ds_write_b32 [[CASTPTR]], v[[K]]

; HSA:  .amdhsa_user_sgpr_private_segment_buffer 1
; HSA:  .amdhsa_user_sgpr_dispatch_ptr 0
; HSA:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @use_flat_to_group_addrspacecast(ptr %ptr) #0 {
  %ftos = addrspacecast ptr %ptr to ptr addrspace(3)
  store volatile i32 0, ptr addrspace(3) %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_private_addrspacecast:

; HSA: s_load_dwordx2 s[[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]]
; CI-DAG v_cmp_ne_u64_e64 vcc, s[[[PTR_LO]]:[[PTR_HI]]], 0{{$}}
; CI-DAG v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], s[[PTR_LO]]
; CI-DAG v_cndmask_b32_e32 [[CASTPTR:v[0-9]+]], -1, v[[VPTR_LO]]
; CI-DAG: v_cmp_ne_u64_e64 s[[[CMP_LO:[0-9]+]]:[[CMP_HI:[0-9]+]]], s[[[PTR_LO]]:[[PTR_HI]]], 0{{$}}
; CI-DAG: s_and_b64 s{{[[0-9]+:[0-9]+]}}, s[[[CMP_LO]]:[[CMP_HI]]], exec
; CI-DAG: s_cselect_b32 [[CASTPTR:s[0-9]+]], s[[PTR_LO]], -1
; CI-DAG: v_mov_b32_e32 [[VCASTPTR:v[0-9]+]], [[CASTPTR]]
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 0{{$}}
; GFX9-DAG: s_cmp_lg_u64 s[[[CMP_LO:[0-9]+]]:[[CMP_HI:[0-9]+]]], 0
; GFX9-DAG: s_cselect_b32 s[[PTR_LO]], s[[PTR_LO]], -1
; GFX9-DAG: v_mov_b32_e32 [[CASTPTR:v[0-9]+]], s[[PTR_LO]]
; CI: buffer_store_dword v[[K]], [[VCASTPTR]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen{{$}}
; GFX9: buffer_store_dword v[[K]], [[CASTPTR]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen{{$}}

; HSA:  .amdhsa_user_sgpr_private_segment_buffer 1
; HSA:  .amdhsa_user_sgpr_dispatch_ptr 0
; HSA:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @use_flat_to_private_addrspacecast(ptr %ptr) #0 {
  %ftos = addrspacecast ptr %ptr to ptr addrspace(5)
  store volatile i32 0, ptr addrspace(5) %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_global_addrspacecast:

; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]], s[4:5], 0x0
; CI-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; CI-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; CI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0
; CI: flat_store_dword v[[[VPTRLO]]:[[VPTRHI]]], [[K]]

; GFX9: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GFX9: global_store_dword [[ZERO]], [[ZERO]], s[[[PTRLO]]:[[PTRHI]]{{\]$}}

; HSA:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @use_flat_to_global_addrspacecast(ptr %ptr) #0 {
  %ftos = addrspacecast ptr %ptr to ptr addrspace(1)
  store volatile i32 0, ptr addrspace(1) %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_constant_addrspacecast:

; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]], s[4:5], 0x0
; HSA: s_load_dword s{{[0-9]+}}, s[[[PTRLO]]:[[PTRHI]]], 0x0

; HSA:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @use_flat_to_constant_addrspacecast(ptr %ptr) #0 {
  %ftos = addrspacecast ptr %ptr to ptr addrspace(4)
  load volatile i32, ptr addrspace(4) %ftos
  ret void
}

; HSA-LABEL: {{^}}cast_0_group_to_flat_addrspacecast:
; CI: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10
; CI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE]]

; GFX9-DAG: s_mov_b64 s[{{[0-9]+}}:[[HI:[0-9]+]]], src_shared_base

; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: {{flat|global}}_store_dword v[[[LO]]:[[HI]]], v[[K]]
define amdgpu_kernel void @cast_0_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast ptr addrspace(3) null to ptr
  store volatile i32 7, ptr %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define amdgpu_kernel void @cast_0_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast ptr null to ptr addrspace(3)
  store volatile i32 7, ptr addrspace(3) %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_group_to_flat_addrspacecast:
; HSA: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA: {{flat|global}}_store_dword v[[[LO]]:[[HI]]], v[[K]]
define amdgpu_kernel void @cast_neg1_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast ptr addrspace(3) inttoptr (i32 -1 to ptr addrspace(3)) to ptr
  store volatile i32 7, ptr %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define amdgpu_kernel void @cast_neg1_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast ptr inttoptr (i64 -1 to ptr) to ptr addrspace(3)
  store volatile i32 7, ptr addrspace(3) %cast
  ret void
}

; FIXME: Shouldn't need to enable queue ptr
; HSA-LABEL: {{^}}cast_0_private_to_flat_addrspacecast:
; CI: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x11
; CI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE]]

; GFX9-DAG: s_mov_b64 s[{{[0-9]+}}:[[HI:[0-9]+]]], src_private_base

; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: {{flat|global}}_store_dword v[[[LO]]:[[HI]]], v[[K]]
define amdgpu_kernel void @cast_0_private_to_flat_addrspacecast() #0 {
  %cast = addrspacecast ptr addrspace(5) null to ptr
  store volatile i32 7, ptr %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_private_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: buffer_store_dword [[K]], [[PTR]], s{{\[[0-9]+:[0-9]+\]}}, 0
define amdgpu_kernel void @cast_0_flat_to_private_addrspacecast() #0 {
  %cast = addrspacecast ptr null to ptr addrspace(5)
  store volatile i32 7, ptr addrspace(5) %cast
  ret void
}


; HSA-LABEL: {{^}}cast_neg1_private_to_flat_addrspacecast:

; HSA: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA: {{flat|global}}_store_dword v[[[LO]]:[[HI]]], v[[K]]

; CI:  .amdhsa_user_sgpr_queue_ptr 1
; GFX9:  .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @cast_neg1_private_to_flat_addrspacecast() #0 {
  %cast = addrspacecast ptr addrspace(5) inttoptr (i32 -1 to ptr addrspace(5)) to ptr
  store volatile i32 7, ptr %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_flat_to_private_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: buffer_store_dword [[K]], [[PTR]], s{{\[[0-9]+:[0-9]+\]}}, 0
define amdgpu_kernel void @cast_neg1_flat_to_private_addrspacecast() #0 {
  %cast = addrspacecast ptr inttoptr (i64 -1 to ptr) to ptr addrspace(5)
  store volatile i32 7, ptr addrspace(5) %cast
  ret void
}


; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.

; HSA-LABEL: {{^}}branch_use_flat_i32:
; HSA: {{flat|global}}_store_dword {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}
; HSA: s_endpgm
define amdgpu_kernel void @branch_use_flat_i32(ptr addrspace(1) noalias %out, ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 %x, i32 %c) #0 {
entry:
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %local, label %global

local:
  %flat_local = addrspacecast ptr addrspace(3) %lptr to ptr
  br label %end

global:
  %flat_global = addrspacecast ptr addrspace(1) %gptr to ptr
  br label %end

end:
  %fptr = phi ptr [ %flat_local, %local ], [ %flat_global, %global ]
  store volatile i32 %x, ptr %fptr, align 4
;  %val = load i32, ptr %fptr, align 4
;  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; Check for prologue initializing special SGPRs pointing to scratch.
; HSA-LABEL: {{^}}store_flat_scratch:
; CI-DAG: s_mov_b32 flat_scratch_lo, s9
; CI-DAG: s_add_i32 [[ADD:s[0-9]+]], s8, s11
; CI-DAG: s_lshr_b32 flat_scratch_hi, [[ADD]], 8

; GFX9: s_add_u32 flat_scratch_lo, s6, s9
; GFX9: s_addc_u32 flat_scratch_hi, s7, 0

; HSA: {{flat|global}}_store_dword
; HSA: s_barrier
; HSA: {{flat|global}}_load_dword
define amdgpu_kernel void @store_flat_scratch(ptr addrspace(1) noalias %out, i32) #0 {
  %alloca = alloca i32, i32 9, align 4, addrspace(5)
  %x = call i32 @llvm.amdgcn.workitem.id.x() #2
  %pptr = getelementptr i32, ptr addrspace(5) %alloca, i32 %x
  %fptr = addrspacecast ptr addrspace(5) %pptr to ptr
  store volatile i32 %x, ptr %fptr
  ; Dummy call
  call void @llvm.amdgcn.s.barrier() #1
  %reload = load volatile i32, ptr %fptr, align 4
  store volatile i32 %reload, ptr addrspace(1) %out, align 4
  ret void
}

; HSA-LABEL: {{^}}use_constant_to_constant32_addrspacecast
; GFX9: s_load_dwordx2 [[PTRPTR:s\[[0-9]+:[0-9]+\]]], s[4:5], 0x0{{$}}
; GFX9: s_load_dword [[OFFSET:s[0-9]+]], s[4:5], 0x8{{$}}
; GFX9: s_load_dwordx2 s[[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]], [[PTRPTR]], 0x0{{$}}
; GFX9: s_mov_b32 s[[PTR_HI]], 0{{$}}
; GFX9: s_add_i32 s[[PTR_LO]], s[[PTR_LO]], [[OFFSET]]
; GFX9: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x0{{$}}
define amdgpu_kernel void @use_constant_to_constant32_addrspacecast(ptr addrspace(4) %ptr.ptr, i32 %offset) #0 {
  %ptr = load volatile ptr addrspace(4), ptr addrspace(4) %ptr.ptr
  %addrspacecast = addrspacecast ptr addrspace(4) %ptr to ptr addrspace(6)
  %gep = getelementptr i8, ptr addrspace(6) %addrspacecast, i32 %offset
  %load = load volatile i32, ptr addrspace(6) %gep, align 4
  ret void
}

; HSA-LABEL: {{^}}use_global_to_constant32_addrspacecast
; GFX9: s_load_dwordx2 [[PTRPTR:s\[[0-9]+:[0-9]+\]]], s[4:5], 0x0{{$}}
; GFX9: s_load_dword [[OFFSET:s[0-9]+]], s[4:5], 0x8{{$}}
; GFX9: s_load_dwordx2 s[[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]], [[PTRPTR]], 0x0{{$}}
; GFX9: s_mov_b32 s[[PTR_HI]], 0{{$}}
; GFX9: s_add_i32 s[[PTR_LO]], s[[PTR_LO]], [[OFFSET]]
; GFX9: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x0{{$}}
define amdgpu_kernel void @use_global_to_constant32_addrspacecast(ptr addrspace(4) %ptr.ptr, i32 %offset) #0 {
  %ptr = load volatile ptr addrspace(1), ptr addrspace(4) %ptr.ptr
  %addrspacecast = addrspacecast ptr addrspace(1) %ptr to ptr addrspace(6)
  %gep = getelementptr i8, ptr addrspace(6) %addrspacecast, i32 %offset
  %load = load volatile i32, ptr addrspace(6) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}use_constant32bit_to_flat_addrspacecast_0:
; GCN: s_load_dword [[PTR:s[0-9]+]],
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], [[PTR]]
; GCN: flat_load_dword v{{[0-9]+}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @use_constant32bit_to_flat_addrspacecast_0(ptr addrspace(6) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(6) %ptr to ptr
  %load = load volatile i32, ptr %stof
  ret void
}

; GCN-LABEL: {{^}}use_constant32bit_to_flat_addrspacecast_1:
; GCN: s_load_dword [[PTR:s[0-9]+]],
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0xffff8000
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], [[PTR]]
; GCN: flat_load_dword v{{[0-9]+}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @use_constant32bit_to_flat_addrspacecast_1(ptr addrspace(6) %ptr) #3 {
  %stof = addrspacecast ptr addrspace(6) %ptr to ptr
  %load = load volatile i32, ptr %stof
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind "amdgpu-32bit-address-high-bits"="0xffff8000" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
