; RUN: opt -passes=amdgpu-attributor -mcpu=kaveri -mattr=-promote-alloca < %s | llc | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=CI %s
; RUN: opt -passes=amdgpu-attributor -mcpu=gfx900 -mattr=-promote-alloca < %s | llc | FileCheck -enable-var-scope -check-prefix=HSA -check-prefix=GFX9 %s

target triple = "amdgcn-amd-amdhsa"

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
; HSA-LABEL: {{^}}use_constant_to_flat_addrspacecast:
; HSA: s_load_dwordx2 s[[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA: flat_load_dword v{{[0-9]+}}, v[[[VPTRLO]]:[[VPTRHI]]]
define amdgpu_kernel void @use_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) #0 {
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  %ld = load volatile i32, ptr %stof
  ret void
}

; HSA-LABEL: {{^}}use_constant_to_global_addrspacecast:
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

; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: flat_store_dword v[[[LO]]:[[HI]]], v[[K]]
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
; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: flat_store_dword v[[[LO]]:[[HI]]], v[[K]]
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

define <2 x ptr addrspace(5)> @addrspacecast_v2p0_to_v2p5(<2 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v2p0_to_v2p5:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr> %ptr to <2 x ptr addrspace(5)>
  ret <2 x ptr addrspace(5)> %cast
}

define <3 x ptr addrspace(5)> @addrspacecast_v3p0_to_v3p5(<3 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v3p0_to_v3p5:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr> %ptr to <3 x ptr addrspace(5)>
  ret <3 x ptr addrspace(5)> %cast
}

define <4 x ptr addrspace(5)> @addrspacecast_v4p0_to_v4p5(<4 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v4p0_to_v4p5:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr> %ptr to <4 x ptr addrspace(5)>
  ret <4 x ptr addrspace(5)> %cast
}

define <8 x ptr addrspace(5)> @addrspacecast_v8p0_to_v8p5(<8 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v8p0_to_v8p5:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v4, -1, v8, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[10:11]
; HSA-NEXT:    v_cndmask_b32_e32 v5, -1, v10, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[12:13]
; HSA-NEXT:    v_cndmask_b32_e32 v6, -1, v12, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[14:15]
; HSA-NEXT:    v_cndmask_b32_e32 v7, -1, v14, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr> %ptr to <8 x ptr addrspace(5)>
  ret <8 x ptr addrspace(5)> %cast
}

define <16 x ptr addrspace(5)> @addrspacecast_v16p0_to_v16p5(<16 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v16p0_to_v16p5:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    buffer_load_dword v31, off, s[0:3], s32
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[4:5], 0, v[24:25]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[6:7], 0, v[26:27]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[8:9], 0, v[28:29]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v4, -1, v8, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[10:11]
; HSA-NEXT:    v_cndmask_b32_e32 v5, -1, v10, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[12:13]
; HSA-NEXT:    v_cndmask_b32_e64 v13, -1, v26, s[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v6, -1, v12, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[14:15]
; HSA-NEXT:    v_cndmask_b32_e64 v12, -1, v24, s[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v7, -1, v14, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[16:17]
; HSA-NEXT:    v_cndmask_b32_e64 v14, -1, v28, s[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v8, -1, v16, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[18:19]
; HSA-NEXT:    v_cndmask_b32_e32 v9, -1, v18, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[20:21]
; HSA-NEXT:    v_cndmask_b32_e32 v10, -1, v20, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[22:23]
; HSA-NEXT:    v_cndmask_b32_e32 v11, -1, v22, vcc
; HSA-NEXT:    s_waitcnt vmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[30:31]
; HSA-NEXT:    v_cndmask_b32_e32 v15, -1, v30, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr> %ptr to <16 x ptr addrspace(5)>
  ret <16 x ptr addrspace(5)> %cast
}

define <2 x ptr> @addrspacecast_v2p5_to_v2p0(<2 x ptr addrspace(5)> %ptr) {
; CI-LABEL: addrspacecast_v2p5_to_v2p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x11
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v3, s4
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v3, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v2, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v3, 0, v3, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v4
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v2p5_to_v2p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v3, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v3, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v2, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v3, 0, v3, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v4
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr addrspace(5)> %ptr to <2 x ptr>
  ret <2 x ptr> %cast
}

define <3 x ptr> @addrspacecast_v3p5_to_v3p0(<3 x ptr addrspace(5)> %ptr) {
; CI-LABEL: addrspacecast_v3p5_to_v3p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x11
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v5, s4
; CI-NEXT:    v_cndmask_b32_e32 v7, 0, v5, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v6, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v3, 0, v5, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v5, 0, v5, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v7
; CI-NEXT:    v_mov_b32_e32 v2, v6
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v3p5_to_v3p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v7, 0, v5, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v6, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v3, 0, v5, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v5, 0, v5, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v7
; GFX9-NEXT:    v_mov_b32_e32 v2, v6
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr addrspace(5)> %ptr to <3 x ptr>
  ret <3 x ptr> %cast
}

define <4 x ptr> @addrspacecast_v4p5_to_v4p0(<4 x ptr addrspace(5)> %ptr) {
; CI-LABEL: addrspacecast_v4p5_to_v4p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x11
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v7, s4
; CI-NEXT:    v_cndmask_b32_e32 v10, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v8, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v5, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v6, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v10
; CI-NEXT:    v_mov_b32_e32 v2, v8
; CI-NEXT:    v_mov_b32_e32 v3, v9
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v4p5_to_v4p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v10, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v8, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v5, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v6, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v10
; GFX9-NEXT:    v_mov_b32_e32 v2, v8
; GFX9-NEXT:    v_mov_b32_e32 v3, v9
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr addrspace(5)> %ptr to <4 x ptr>
  ret <4 x ptr> %cast
}

define <8 x ptr> @addrspacecast_v8p5_to_v8p0(<8 x ptr addrspace(5)> %ptr) {
; CI-LABEL: addrspacecast_v8p5_to_v8p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x11
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v15, s4
; CI-NEXT:    v_cndmask_b32_e32 v22, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v16, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v17, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v18, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v19, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v20, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v21, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; CI-NEXT:    v_cndmask_b32_e32 v8, 0, v4, vcc
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v5
; CI-NEXT:    v_cndmask_b32_e32 v10, 0, v5, vcc
; CI-NEXT:    v_cndmask_b32_e32 v11, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v6
; CI-NEXT:    v_cndmask_b32_e32 v12, 0, v6, vcc
; CI-NEXT:    v_cndmask_b32_e32 v13, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v7
; CI-NEXT:    v_cndmask_b32_e32 v14, 0, v7, vcc
; CI-NEXT:    v_cndmask_b32_e32 v15, 0, v15, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v22
; CI-NEXT:    v_mov_b32_e32 v2, v16
; CI-NEXT:    v_mov_b32_e32 v3, v17
; CI-NEXT:    v_mov_b32_e32 v4, v18
; CI-NEXT:    v_mov_b32_e32 v5, v19
; CI-NEXT:    v_mov_b32_e32 v6, v20
; CI-NEXT:    v_mov_b32_e32 v7, v21
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v8p5_to_v8p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v15, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v22, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v16, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v17, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v18, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v19, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v20, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v21, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; GFX9-NEXT:    v_cndmask_b32_e32 v8, 0, v4, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v5
; GFX9-NEXT:    v_cndmask_b32_e32 v10, 0, v5, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v11, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v6
; GFX9-NEXT:    v_cndmask_b32_e32 v12, 0, v6, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v13, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v7
; GFX9-NEXT:    v_cndmask_b32_e32 v14, 0, v7, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v15, 0, v15, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v22
; GFX9-NEXT:    v_mov_b32_e32 v2, v16
; GFX9-NEXT:    v_mov_b32_e32 v3, v17
; GFX9-NEXT:    v_mov_b32_e32 v4, v18
; GFX9-NEXT:    v_mov_b32_e32 v5, v19
; GFX9-NEXT:    v_mov_b32_e32 v6, v20
; GFX9-NEXT:    v_mov_b32_e32 v7, v21
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr addrspace(5)> %ptr to <8 x ptr>
  ret <8 x ptr> %cast
}

define <16 x ptr> @addrspacecast_v16p5_to_v16p0(<16 x ptr addrspace(5)> %ptr) {
; CI-LABEL: addrspacecast_v16p5_to_v16p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x11
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    v_cmp_ne_u32_e64 s[6:7], -1, v6
; CI-NEXT:    v_cmp_ne_u32_e64 s[8:9], -1, v7
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v31, s4
; CI-NEXT:    v_cndmask_b32_e32 v48, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v35, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v33, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v36, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v49, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v37, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v34, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; CI-NEXT:    v_cmp_ne_u32_e64 s[4:5], -1, v5
; CI-NEXT:    v_cndmask_b32_e32 v38, 0, v4, vcc
; CI-NEXT:    v_cndmask_b32_e64 v50, 0, v5, s[4:5]
; CI-NEXT:    v_cndmask_b32_e64 v39, 0, v6, s[6:7]
; CI-NEXT:    v_cndmask_b32_e64 v32, 0, v7, s[8:9]
; CI-NEXT:    v_cmp_ne_u32_e64 s[10:11], -1, v8
; CI-NEXT:    v_cmp_ne_u32_e64 s[12:13], -1, v9
; CI-NEXT:    v_cmp_ne_u32_e64 s[14:15], -1, v10
; CI-NEXT:    v_cmp_ne_u32_e64 s[16:17], -1, v11
; CI-NEXT:    v_cmp_ne_u32_e64 s[18:19], -1, v12
; CI-NEXT:    v_cmp_ne_u32_e64 s[20:21], -1, v13
; CI-NEXT:    v_cmp_ne_u32_e64 s[22:23], -1, v14
; CI-NEXT:    v_cmp_ne_u32_e64 s[24:25], -1, v15
; CI-NEXT:    v_cndmask_b32_e64 v16, 0, v8, s[10:11]
; CI-NEXT:    v_cndmask_b32_e64 v18, 0, v9, s[12:13]
; CI-NEXT:    v_cndmask_b32_e64 v20, 0, v10, s[14:15]
; CI-NEXT:    v_cndmask_b32_e64 v22, 0, v11, s[16:17]
; CI-NEXT:    v_cndmask_b32_e64 v24, 0, v12, s[18:19]
; CI-NEXT:    v_cndmask_b32_e64 v26, 0, v13, s[20:21]
; CI-NEXT:    v_cndmask_b32_e64 v28, 0, v14, s[22:23]
; CI-NEXT:    v_cndmask_b32_e64 v30, 0, v15, s[24:25]
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v31, vcc
; CI-NEXT:    v_cndmask_b32_e64 v11, 0, v31, s[4:5]
; CI-NEXT:    v_cndmask_b32_e64 v13, 0, v31, s[6:7]
; CI-NEXT:    v_cndmask_b32_e64 v15, 0, v31, s[8:9]
; CI-NEXT:    v_cndmask_b32_e64 v17, 0, v31, s[10:11]
; CI-NEXT:    v_cndmask_b32_e64 v19, 0, v31, s[12:13]
; CI-NEXT:    v_cndmask_b32_e64 v21, 0, v31, s[14:15]
; CI-NEXT:    v_cndmask_b32_e64 v23, 0, v31, s[16:17]
; CI-NEXT:    v_cndmask_b32_e64 v25, 0, v31, s[18:19]
; CI-NEXT:    v_cndmask_b32_e64 v27, 0, v31, s[20:21]
; CI-NEXT:    v_cndmask_b32_e64 v29, 0, v31, s[22:23]
; CI-NEXT:    v_cndmask_b32_e64 v31, 0, v31, s[24:25]
; CI-NEXT:    v_mov_b32_e32 v1, v48
; CI-NEXT:    v_mov_b32_e32 v2, v35
; CI-NEXT:    v_mov_b32_e32 v3, v33
; CI-NEXT:    v_mov_b32_e32 v4, v36
; CI-NEXT:    v_mov_b32_e32 v5, v49
; CI-NEXT:    v_mov_b32_e32 v6, v37
; CI-NEXT:    v_mov_b32_e32 v7, v34
; CI-NEXT:    v_mov_b32_e32 v8, v38
; CI-NEXT:    v_mov_b32_e32 v10, v50
; CI-NEXT:    v_mov_b32_e32 v12, v39
; CI-NEXT:    v_mov_b32_e32 v14, v32
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v16p5_to_v16p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v31, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v48, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v35, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v33, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v36, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v49, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v37, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v34, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[4:5], -1, v5
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[6:7], -1, v6
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[8:9], -1, v7
; GFX9-NEXT:    v_cndmask_b32_e32 v38, 0, v4, vcc
; GFX9-NEXT:    v_cndmask_b32_e64 v50, 0, v5, s[4:5]
; GFX9-NEXT:    v_cndmask_b32_e64 v39, 0, v6, s[6:7]
; GFX9-NEXT:    v_cndmask_b32_e64 v32, 0, v7, s[8:9]
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[10:11], -1, v8
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[12:13], -1, v9
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[14:15], -1, v10
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[16:17], -1, v11
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[18:19], -1, v12
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[20:21], -1, v13
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[22:23], -1, v14
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[24:25], -1, v15
; GFX9-NEXT:    v_cndmask_b32_e64 v16, 0, v8, s[10:11]
; GFX9-NEXT:    v_cndmask_b32_e64 v18, 0, v9, s[12:13]
; GFX9-NEXT:    v_cndmask_b32_e64 v20, 0, v10, s[14:15]
; GFX9-NEXT:    v_cndmask_b32_e64 v22, 0, v11, s[16:17]
; GFX9-NEXT:    v_cndmask_b32_e64 v24, 0, v12, s[18:19]
; GFX9-NEXT:    v_cndmask_b32_e64 v26, 0, v13, s[20:21]
; GFX9-NEXT:    v_cndmask_b32_e64 v28, 0, v14, s[22:23]
; GFX9-NEXT:    v_cndmask_b32_e64 v30, 0, v15, s[24:25]
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v31, vcc
; GFX9-NEXT:    v_cndmask_b32_e64 v11, 0, v31, s[4:5]
; GFX9-NEXT:    v_cndmask_b32_e64 v13, 0, v31, s[6:7]
; GFX9-NEXT:    v_cndmask_b32_e64 v15, 0, v31, s[8:9]
; GFX9-NEXT:    v_cndmask_b32_e64 v17, 0, v31, s[10:11]
; GFX9-NEXT:    v_cndmask_b32_e64 v19, 0, v31, s[12:13]
; GFX9-NEXT:    v_cndmask_b32_e64 v21, 0, v31, s[14:15]
; GFX9-NEXT:    v_cndmask_b32_e64 v23, 0, v31, s[16:17]
; GFX9-NEXT:    v_cndmask_b32_e64 v25, 0, v31, s[18:19]
; GFX9-NEXT:    v_cndmask_b32_e64 v27, 0, v31, s[20:21]
; GFX9-NEXT:    v_cndmask_b32_e64 v29, 0, v31, s[22:23]
; GFX9-NEXT:    v_cndmask_b32_e64 v31, 0, v31, s[24:25]
; GFX9-NEXT:    v_mov_b32_e32 v1, v48
; GFX9-NEXT:    v_mov_b32_e32 v2, v35
; GFX9-NEXT:    v_mov_b32_e32 v3, v33
; GFX9-NEXT:    v_mov_b32_e32 v4, v36
; GFX9-NEXT:    v_mov_b32_e32 v5, v49
; GFX9-NEXT:    v_mov_b32_e32 v6, v37
; GFX9-NEXT:    v_mov_b32_e32 v7, v34
; GFX9-NEXT:    v_mov_b32_e32 v8, v38
; GFX9-NEXT:    v_mov_b32_e32 v10, v50
; GFX9-NEXT:    v_mov_b32_e32 v12, v39
; GFX9-NEXT:    v_mov_b32_e32 v14, v32
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr addrspace(5)> %ptr to <16 x ptr>
  ret <16 x ptr> %cast
}

define <2 x ptr addrspace(3)> @addrspacecast_v2p0_to_v2p3(<2 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v2p0_to_v2p3:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr> %ptr to <2 x ptr addrspace(3)>
  ret <2 x ptr addrspace(3)> %cast
}

define <3 x ptr addrspace(3)> @addrspacecast_v3p0_to_v3p3(<3 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v3p0_to_v3p3:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr> %ptr to <3 x ptr addrspace(3)>
  ret <3 x ptr addrspace(3)> %cast
}

define <4 x ptr addrspace(3)> @addrspacecast_v4p0_to_v4p3(<4 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v4p0_to_v4p3:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr> %ptr to <4 x ptr addrspace(3)>
  ret <4 x ptr addrspace(3)> %cast
}

define <8 x ptr addrspace(3)> @addrspacecast_v8p0_to_v8p3(<8 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v8p0_to_v8p3:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v4, -1, v8, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[10:11]
; HSA-NEXT:    v_cndmask_b32_e32 v5, -1, v10, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[12:13]
; HSA-NEXT:    v_cndmask_b32_e32 v6, -1, v12, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[14:15]
; HSA-NEXT:    v_cndmask_b32_e32 v7, -1, v14, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr> %ptr to <8 x ptr addrspace(3)>
  ret <8 x ptr addrspace(3)> %cast
}

define <16 x ptr addrspace(3)> @addrspacecast_v16p0_to_v16p3(<16 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v16p0_to_v16p3:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    buffer_load_dword v31, off, s[0:3], s32
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[0:1]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[4:5], 0, v[24:25]
; HSA-NEXT:    v_cndmask_b32_e32 v0, -1, v0, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[2:3]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[6:7], 0, v[26:27]
; HSA-NEXT:    v_cndmask_b32_e32 v1, -1, v2, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[4:5]
; HSA-NEXT:    v_cmp_ne_u64_e64 s[8:9], 0, v[28:29]
; HSA-NEXT:    v_cndmask_b32_e32 v2, -1, v4, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v3, -1, v6, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v4, -1, v8, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[10:11]
; HSA-NEXT:    v_cndmask_b32_e32 v5, -1, v10, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[12:13]
; HSA-NEXT:    v_cndmask_b32_e64 v13, -1, v26, s[6:7]
; HSA-NEXT:    v_cndmask_b32_e32 v6, -1, v12, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[14:15]
; HSA-NEXT:    v_cndmask_b32_e64 v12, -1, v24, s[4:5]
; HSA-NEXT:    v_cndmask_b32_e32 v7, -1, v14, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[16:17]
; HSA-NEXT:    v_cndmask_b32_e64 v14, -1, v28, s[8:9]
; HSA-NEXT:    v_cndmask_b32_e32 v8, -1, v16, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[18:19]
; HSA-NEXT:    v_cndmask_b32_e32 v9, -1, v18, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[20:21]
; HSA-NEXT:    v_cndmask_b32_e32 v10, -1, v20, vcc
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[22:23]
; HSA-NEXT:    v_cndmask_b32_e32 v11, -1, v22, vcc
; HSA-NEXT:    s_waitcnt vmcnt(0)
; HSA-NEXT:    v_cmp_ne_u64_e32 vcc, 0, v[30:31]
; HSA-NEXT:    v_cndmask_b32_e32 v15, -1, v30, vcc
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr> %ptr to <16 x ptr addrspace(3)>
  ret <16 x ptr addrspace(3)> %cast
}

define <2 x ptr> @addrspacecast_v2p3_to_v2p0(<2 x ptr addrspace(3)> %ptr) {
; CI-LABEL: addrspacecast_v2p3_to_v2p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x10
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v3, s4
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v3, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v2, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v3, 0, v3, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v4
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v2p3_to_v2p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_shared_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v3, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v3, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v2, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v3, 0, v3, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v4
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr addrspace(3)> %ptr to <2 x ptr>
  ret <2 x ptr> %cast
}

define <3 x ptr> @addrspacecast_v3p3_to_v3p0(<3 x ptr addrspace(3)> %ptr) {
; CI-LABEL: addrspacecast_v3p3_to_v3p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x10
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v5, s4
; CI-NEXT:    v_cndmask_b32_e32 v7, 0, v5, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v6, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v3, 0, v5, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v5, 0, v5, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v7
; CI-NEXT:    v_mov_b32_e32 v2, v6
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v3p3_to_v3p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_shared_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v7, 0, v5, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v6, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v3, 0, v5, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v5, 0, v5, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v7
; GFX9-NEXT:    v_mov_b32_e32 v2, v6
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr addrspace(3)> %ptr to <3 x ptr>
  ret <3 x ptr> %cast
}

define <4 x ptr> @addrspacecast_v4p3_to_v4p0(<4 x ptr addrspace(3)> %ptr) {
; CI-LABEL: addrspacecast_v4p3_to_v4p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x10
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v7, s4
; CI-NEXT:    v_cndmask_b32_e32 v10, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v8, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v5, 0, v7, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v6, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v10
; CI-NEXT:    v_mov_b32_e32 v2, v8
; CI-NEXT:    v_mov_b32_e32 v3, v9
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v4p3_to_v4p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_shared_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v10, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v8, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v4, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v5, 0, v7, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v6, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v7, 0, v7, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v10
; GFX9-NEXT:    v_mov_b32_e32 v2, v8
; GFX9-NEXT:    v_mov_b32_e32 v3, v9
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr addrspace(3)> %ptr to <4 x ptr>
  ret <4 x ptr> %cast
}

define <8 x ptr> @addrspacecast_v8p3_to_v8p0(<8 x ptr addrspace(3)> %ptr) {
; CI-LABEL: addrspacecast_v8p3_to_v8p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x10
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v15, s4
; CI-NEXT:    v_cndmask_b32_e32 v22, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v16, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v17, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v18, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v19, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v20, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v21, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; CI-NEXT:    v_cndmask_b32_e32 v8, 0, v4, vcc
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v5
; CI-NEXT:    v_cndmask_b32_e32 v10, 0, v5, vcc
; CI-NEXT:    v_cndmask_b32_e32 v11, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v6
; CI-NEXT:    v_cndmask_b32_e32 v12, 0, v6, vcc
; CI-NEXT:    v_cndmask_b32_e32 v13, 0, v15, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v7
; CI-NEXT:    v_cndmask_b32_e32 v14, 0, v7, vcc
; CI-NEXT:    v_cndmask_b32_e32 v15, 0, v15, vcc
; CI-NEXT:    v_mov_b32_e32 v1, v22
; CI-NEXT:    v_mov_b32_e32 v2, v16
; CI-NEXT:    v_mov_b32_e32 v3, v17
; CI-NEXT:    v_mov_b32_e32 v4, v18
; CI-NEXT:    v_mov_b32_e32 v5, v19
; CI-NEXT:    v_mov_b32_e32 v6, v20
; CI-NEXT:    v_mov_b32_e32 v7, v21
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v8p3_to_v8p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_shared_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v15, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v22, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v16, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v17, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v18, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v19, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v20, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v21, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; GFX9-NEXT:    v_cndmask_b32_e32 v8, 0, v4, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v5
; GFX9-NEXT:    v_cndmask_b32_e32 v10, 0, v5, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v11, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v6
; GFX9-NEXT:    v_cndmask_b32_e32 v12, 0, v6, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v13, 0, v15, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v7
; GFX9-NEXT:    v_cndmask_b32_e32 v14, 0, v7, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v15, 0, v15, vcc
; GFX9-NEXT:    v_mov_b32_e32 v1, v22
; GFX9-NEXT:    v_mov_b32_e32 v2, v16
; GFX9-NEXT:    v_mov_b32_e32 v3, v17
; GFX9-NEXT:    v_mov_b32_e32 v4, v18
; GFX9-NEXT:    v_mov_b32_e32 v5, v19
; GFX9-NEXT:    v_mov_b32_e32 v6, v20
; GFX9-NEXT:    v_mov_b32_e32 v7, v21
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr addrspace(3)> %ptr to <8 x ptr>
  ret <8 x ptr> %cast
}

define <16 x ptr> @addrspacecast_v16p3_to_v16p0(<16 x ptr addrspace(3)> %ptr) {
; CI-LABEL: addrspacecast_v16p3_to_v16p0:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_load_dword s4, s[6:7], 0x10
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; CI-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; CI-NEXT:    v_cmp_ne_u32_e64 s[6:7], -1, v6
; CI-NEXT:    v_cmp_ne_u32_e64 s[8:9], -1, v7
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v31, s4
; CI-NEXT:    v_cndmask_b32_e32 v48, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; CI-NEXT:    v_cndmask_b32_e32 v35, 0, v1, vcc
; CI-NEXT:    v_cndmask_b32_e32 v33, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; CI-NEXT:    v_cndmask_b32_e32 v36, 0, v2, vcc
; CI-NEXT:    v_cndmask_b32_e32 v49, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; CI-NEXT:    v_cndmask_b32_e32 v37, 0, v3, vcc
; CI-NEXT:    v_cndmask_b32_e32 v34, 0, v31, vcc
; CI-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; CI-NEXT:    v_cmp_ne_u32_e64 s[4:5], -1, v5
; CI-NEXT:    v_cndmask_b32_e32 v38, 0, v4, vcc
; CI-NEXT:    v_cndmask_b32_e64 v50, 0, v5, s[4:5]
; CI-NEXT:    v_cndmask_b32_e64 v39, 0, v6, s[6:7]
; CI-NEXT:    v_cndmask_b32_e64 v32, 0, v7, s[8:9]
; CI-NEXT:    v_cmp_ne_u32_e64 s[10:11], -1, v8
; CI-NEXT:    v_cmp_ne_u32_e64 s[12:13], -1, v9
; CI-NEXT:    v_cmp_ne_u32_e64 s[14:15], -1, v10
; CI-NEXT:    v_cmp_ne_u32_e64 s[16:17], -1, v11
; CI-NEXT:    v_cmp_ne_u32_e64 s[18:19], -1, v12
; CI-NEXT:    v_cmp_ne_u32_e64 s[20:21], -1, v13
; CI-NEXT:    v_cmp_ne_u32_e64 s[22:23], -1, v14
; CI-NEXT:    v_cmp_ne_u32_e64 s[24:25], -1, v15
; CI-NEXT:    v_cndmask_b32_e64 v16, 0, v8, s[10:11]
; CI-NEXT:    v_cndmask_b32_e64 v18, 0, v9, s[12:13]
; CI-NEXT:    v_cndmask_b32_e64 v20, 0, v10, s[14:15]
; CI-NEXT:    v_cndmask_b32_e64 v22, 0, v11, s[16:17]
; CI-NEXT:    v_cndmask_b32_e64 v24, 0, v12, s[18:19]
; CI-NEXT:    v_cndmask_b32_e64 v26, 0, v13, s[20:21]
; CI-NEXT:    v_cndmask_b32_e64 v28, 0, v14, s[22:23]
; CI-NEXT:    v_cndmask_b32_e64 v30, 0, v15, s[24:25]
; CI-NEXT:    v_cndmask_b32_e32 v9, 0, v31, vcc
; CI-NEXT:    v_cndmask_b32_e64 v11, 0, v31, s[4:5]
; CI-NEXT:    v_cndmask_b32_e64 v13, 0, v31, s[6:7]
; CI-NEXT:    v_cndmask_b32_e64 v15, 0, v31, s[8:9]
; CI-NEXT:    v_cndmask_b32_e64 v17, 0, v31, s[10:11]
; CI-NEXT:    v_cndmask_b32_e64 v19, 0, v31, s[12:13]
; CI-NEXT:    v_cndmask_b32_e64 v21, 0, v31, s[14:15]
; CI-NEXT:    v_cndmask_b32_e64 v23, 0, v31, s[16:17]
; CI-NEXT:    v_cndmask_b32_e64 v25, 0, v31, s[18:19]
; CI-NEXT:    v_cndmask_b32_e64 v27, 0, v31, s[20:21]
; CI-NEXT:    v_cndmask_b32_e64 v29, 0, v31, s[22:23]
; CI-NEXT:    v_cndmask_b32_e64 v31, 0, v31, s[24:25]
; CI-NEXT:    v_mov_b32_e32 v1, v48
; CI-NEXT:    v_mov_b32_e32 v2, v35
; CI-NEXT:    v_mov_b32_e32 v3, v33
; CI-NEXT:    v_mov_b32_e32 v4, v36
; CI-NEXT:    v_mov_b32_e32 v5, v49
; CI-NEXT:    v_mov_b32_e32 v6, v37
; CI-NEXT:    v_mov_b32_e32 v7, v34
; CI-NEXT:    v_mov_b32_e32 v8, v38
; CI-NEXT:    v_mov_b32_e32 v10, v50
; CI-NEXT:    v_mov_b32_e32 v12, v39
; CI-NEXT:    v_mov_b32_e32 v14, v32
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: addrspacecast_v16p3_to_v16p0:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], src_shared_base
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v0
; GFX9-NEXT:    v_mov_b32_e32 v31, s5
; GFX9-NEXT:    v_cndmask_b32_e32 v0, 0, v0, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v48, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v1
; GFX9-NEXT:    v_cndmask_b32_e32 v35, 0, v1, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v33, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v2
; GFX9-NEXT:    v_cndmask_b32_e32 v36, 0, v2, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v49, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v3
; GFX9-NEXT:    v_cndmask_b32_e32 v37, 0, v3, vcc
; GFX9-NEXT:    v_cndmask_b32_e32 v34, 0, v31, vcc
; GFX9-NEXT:    v_cmp_ne_u32_e32 vcc, -1, v4
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[4:5], -1, v5
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[6:7], -1, v6
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[8:9], -1, v7
; GFX9-NEXT:    v_cndmask_b32_e32 v38, 0, v4, vcc
; GFX9-NEXT:    v_cndmask_b32_e64 v50, 0, v5, s[4:5]
; GFX9-NEXT:    v_cndmask_b32_e64 v39, 0, v6, s[6:7]
; GFX9-NEXT:    v_cndmask_b32_e64 v32, 0, v7, s[8:9]
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[10:11], -1, v8
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[12:13], -1, v9
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[14:15], -1, v10
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[16:17], -1, v11
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[18:19], -1, v12
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[20:21], -1, v13
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[22:23], -1, v14
; GFX9-NEXT:    v_cmp_ne_u32_e64 s[24:25], -1, v15
; GFX9-NEXT:    v_cndmask_b32_e64 v16, 0, v8, s[10:11]
; GFX9-NEXT:    v_cndmask_b32_e64 v18, 0, v9, s[12:13]
; GFX9-NEXT:    v_cndmask_b32_e64 v20, 0, v10, s[14:15]
; GFX9-NEXT:    v_cndmask_b32_e64 v22, 0, v11, s[16:17]
; GFX9-NEXT:    v_cndmask_b32_e64 v24, 0, v12, s[18:19]
; GFX9-NEXT:    v_cndmask_b32_e64 v26, 0, v13, s[20:21]
; GFX9-NEXT:    v_cndmask_b32_e64 v28, 0, v14, s[22:23]
; GFX9-NEXT:    v_cndmask_b32_e64 v30, 0, v15, s[24:25]
; GFX9-NEXT:    v_cndmask_b32_e32 v9, 0, v31, vcc
; GFX9-NEXT:    v_cndmask_b32_e64 v11, 0, v31, s[4:5]
; GFX9-NEXT:    v_cndmask_b32_e64 v13, 0, v31, s[6:7]
; GFX9-NEXT:    v_cndmask_b32_e64 v15, 0, v31, s[8:9]
; GFX9-NEXT:    v_cndmask_b32_e64 v17, 0, v31, s[10:11]
; GFX9-NEXT:    v_cndmask_b32_e64 v19, 0, v31, s[12:13]
; GFX9-NEXT:    v_cndmask_b32_e64 v21, 0, v31, s[14:15]
; GFX9-NEXT:    v_cndmask_b32_e64 v23, 0, v31, s[16:17]
; GFX9-NEXT:    v_cndmask_b32_e64 v25, 0, v31, s[18:19]
; GFX9-NEXT:    v_cndmask_b32_e64 v27, 0, v31, s[20:21]
; GFX9-NEXT:    v_cndmask_b32_e64 v29, 0, v31, s[22:23]
; GFX9-NEXT:    v_cndmask_b32_e64 v31, 0, v31, s[24:25]
; GFX9-NEXT:    v_mov_b32_e32 v1, v48
; GFX9-NEXT:    v_mov_b32_e32 v2, v35
; GFX9-NEXT:    v_mov_b32_e32 v3, v33
; GFX9-NEXT:    v_mov_b32_e32 v4, v36
; GFX9-NEXT:    v_mov_b32_e32 v5, v49
; GFX9-NEXT:    v_mov_b32_e32 v6, v37
; GFX9-NEXT:    v_mov_b32_e32 v7, v34
; GFX9-NEXT:    v_mov_b32_e32 v8, v38
; GFX9-NEXT:    v_mov_b32_e32 v10, v50
; GFX9-NEXT:    v_mov_b32_e32 v12, v39
; GFX9-NEXT:    v_mov_b32_e32 v14, v32
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr addrspace(3)> %ptr to <16 x ptr>
  ret <16 x ptr> %cast
}

define <2 x ptr addrspace(1)> @addrspacecast_v2p0_to_v2p1(<2 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v2p0_to_v2p1:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr> %ptr to <2 x ptr addrspace(1)>
  ret <2 x ptr addrspace(1)> %cast
}

define <3 x ptr addrspace(1)> @addrspacecast_v3p0_to_v3p1(<3 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v3p0_to_v3p1:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr> %ptr to <3 x ptr addrspace(1)>
  ret <3 x ptr addrspace(1)> %cast
}

define <4 x ptr addrspace(1)> @addrspacecast_v4p0_to_v4p1(<4 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v4p0_to_v4p1:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr> %ptr to <4 x ptr addrspace(1)>
  ret <4 x ptr addrspace(1)> %cast
}

define <8 x ptr addrspace(1)> @addrspacecast_v8p0_to_v8p1(<8 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v8p0_to_v8p1:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr> %ptr to <8 x ptr addrspace(1)>
  ret <8 x ptr addrspace(1)> %cast
}

define <16 x ptr addrspace(1)> @addrspacecast_v16p0_to_v16p1(<16 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v16p0_to_v16p1:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    buffer_load_dword v31, off, s[0:3], s32
; HSA-NEXT:    s_waitcnt vmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr> %ptr to <16 x ptr addrspace(1)>
  ret <16 x ptr addrspace(1)> %cast
}

define <2 x ptr> @addrspacecast_v2p1_to_v2p0(<2 x ptr addrspace(1)> %ptr) {
; HSA-LABEL: addrspacecast_v2p1_to_v2p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr addrspace(1)> %ptr to <2 x ptr>
  ret <2 x ptr> %cast
}

define <1 x ptr> @addrspacecast_v1p1_to_v1p0(<1 x ptr addrspace(1)> %ptr) {
; HSA-LABEL: addrspacecast_v1p1_to_v1p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <1 x ptr addrspace(1)> %ptr to <1 x ptr>
  ret <1 x ptr> %cast
}

define <4 x ptr> @addrspacecast_v4p1_to_v4p0(<4 x ptr addrspace(1)> %ptr) {
; HSA-LABEL: addrspacecast_v4p1_to_v4p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr addrspace(1)> %ptr to <4 x ptr>
  ret <4 x ptr> %cast
}

define <8 x ptr> @addrspacecast_v8p1_to_v8p0(<8 x ptr addrspace(1)> %ptr) {
; HSA-LABEL: addrspacecast_v8p1_to_v8p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr addrspace(1)> %ptr to <8 x ptr>
  ret <8 x ptr> %cast
}

define <16 x ptr> @addrspacecast_v16p1_to_v16p0(<16 x ptr addrspace(1)> %ptr) {
; HSA-LABEL: addrspacecast_v16p1_to_v16p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    buffer_load_dword v31, off, s[0:3], s32
; HSA-NEXT:    s_waitcnt vmcnt(0)
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr addrspace(1)> %ptr to <16 x ptr>
  ret <16 x ptr> %cast
}

define <2 x ptr addrspace(6)> @addrspacecast_v2p0_to_v2p6(<2 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v2p0_to_v2p6:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v1, v2
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr> %ptr to <2 x ptr addrspace(6)>
  ret <2 x ptr addrspace(6)> %cast
}

define <3 x ptr addrspace(6)> @addrspacecast_v3p0_to_v3p6(<3 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v3p0_to_v3p6:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v1, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v4
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <3 x ptr> %ptr to <3 x ptr addrspace(6)>
  ret <3 x ptr addrspace(6)> %cast
}

define <4 x ptr addrspace(6)> @addrspacecast_v4p0_to_v4p6(<4 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v4p0_to_v4p6:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v3, v6
; HSA-NEXT:    v_mov_b32_e32 v1, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v4
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr> %ptr to <4 x ptr addrspace(6)>
  ret <4 x ptr addrspace(6)> %cast
}

define <8 x ptr addrspace(6)> @addrspacecast_v8p0_to_v8p6(<8 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v8p0_to_v8p6:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v7, v14
; HSA-NEXT:    v_mov_b32_e32 v5, v10
; HSA-NEXT:    v_mov_b32_e32 v3, v6
; HSA-NEXT:    v_mov_b32_e32 v1, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v4
; HSA-NEXT:    v_mov_b32_e32 v4, v8
; HSA-NEXT:    v_mov_b32_e32 v6, v12
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr> %ptr to <8 x ptr addrspace(6)>
  ret <8 x ptr addrspace(6)> %cast
}

define <16 x ptr addrspace(6)> @addrspacecast_v16p0_to_v16p6(<16 x ptr> %ptr) {
; HSA-LABEL: addrspacecast_v16p0_to_v16p6:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v15, v30
; HSA-NEXT:    v_mov_b32_e32 v13, v26
; HSA-NEXT:    v_mov_b32_e32 v11, v22
; HSA-NEXT:    v_mov_b32_e32 v9, v18
; HSA-NEXT:    v_mov_b32_e32 v7, v14
; HSA-NEXT:    v_mov_b32_e32 v5, v10
; HSA-NEXT:    v_mov_b32_e32 v3, v6
; HSA-NEXT:    v_mov_b32_e32 v1, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v4
; HSA-NEXT:    v_mov_b32_e32 v4, v8
; HSA-NEXT:    v_mov_b32_e32 v6, v12
; HSA-NEXT:    v_mov_b32_e32 v8, v16
; HSA-NEXT:    v_mov_b32_e32 v10, v20
; HSA-NEXT:    v_mov_b32_e32 v12, v24
; HSA-NEXT:    v_mov_b32_e32 v14, v28
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr> %ptr to <16 x ptr addrspace(6)>
  ret <16 x ptr addrspace(6)> %cast
}

define <2 x ptr> @addrspacecast_v2p6_to_v2p0(<2 x ptr addrspace(6)> %ptr) {
; HSA-LABEL: addrspacecast_v2p6_to_v2p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v2, v1
; HSA-NEXT:    v_mov_b32_e32 v1, 0
; HSA-NEXT:    v_mov_b32_e32 v3, 0
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <2 x ptr addrspace(6)> %ptr to <2 x ptr>
  ret <2 x ptr> %cast
}

define <1 x ptr> @addrspacecast_v1p6_to_v1p0(<1 x ptr addrspace(6)> %ptr) {
; HSA-LABEL: addrspacecast_v1p6_to_v1p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v1, 0
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <1 x ptr addrspace(6)> %ptr to <1 x ptr>
  ret <1 x ptr> %cast
}

define <4 x ptr> @addrspacecast_v4p6_to_v4p0(<4 x ptr addrspace(6)> %ptr) {
; HSA-LABEL: addrspacecast_v4p6_to_v4p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v6, v3
; HSA-NEXT:    v_mov_b32_e32 v4, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v1
; HSA-NEXT:    v_mov_b32_e32 v1, 0
; HSA-NEXT:    v_mov_b32_e32 v3, 0
; HSA-NEXT:    v_mov_b32_e32 v5, 0
; HSA-NEXT:    v_mov_b32_e32 v7, 0
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <4 x ptr addrspace(6)> %ptr to <4 x ptr>
  ret <4 x ptr> %cast
}

define <8 x ptr> @addrspacecast_v8p6_to_v8p0(<8 x ptr addrspace(6)> %ptr) {
; HSA-LABEL: addrspacecast_v8p6_to_v8p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v14, v7
; HSA-NEXT:    v_mov_b32_e32 v12, v6
; HSA-NEXT:    v_mov_b32_e32 v10, v5
; HSA-NEXT:    v_mov_b32_e32 v8, v4
; HSA-NEXT:    v_mov_b32_e32 v6, v3
; HSA-NEXT:    v_mov_b32_e32 v4, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v1
; HSA-NEXT:    v_mov_b32_e32 v1, 0
; HSA-NEXT:    v_mov_b32_e32 v3, 0
; HSA-NEXT:    v_mov_b32_e32 v5, 0
; HSA-NEXT:    v_mov_b32_e32 v7, 0
; HSA-NEXT:    v_mov_b32_e32 v9, 0
; HSA-NEXT:    v_mov_b32_e32 v11, 0
; HSA-NEXT:    v_mov_b32_e32 v13, 0
; HSA-NEXT:    v_mov_b32_e32 v15, 0
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <8 x ptr addrspace(6)> %ptr to <8 x ptr>
  ret <8 x ptr> %cast
}

define <16 x ptr> @addrspacecast_v16p6_to_v16p0(<16 x ptr addrspace(6)> %ptr) {
; HSA-LABEL: addrspacecast_v16p6_to_v16p0:
; HSA:       ; %bb.0:
; HSA-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; HSA-NEXT:    v_mov_b32_e32 v30, v15
; HSA-NEXT:    v_mov_b32_e32 v28, v14
; HSA-NEXT:    v_mov_b32_e32 v26, v13
; HSA-NEXT:    v_mov_b32_e32 v24, v12
; HSA-NEXT:    v_mov_b32_e32 v22, v11
; HSA-NEXT:    v_mov_b32_e32 v20, v10
; HSA-NEXT:    v_mov_b32_e32 v18, v9
; HSA-NEXT:    v_mov_b32_e32 v16, v8
; HSA-NEXT:    v_mov_b32_e32 v14, v7
; HSA-NEXT:    v_mov_b32_e32 v12, v6
; HSA-NEXT:    v_mov_b32_e32 v10, v5
; HSA-NEXT:    v_mov_b32_e32 v8, v4
; HSA-NEXT:    v_mov_b32_e32 v6, v3
; HSA-NEXT:    v_mov_b32_e32 v4, v2
; HSA-NEXT:    v_mov_b32_e32 v2, v1
; HSA-NEXT:    v_mov_b32_e32 v1, 0
; HSA-NEXT:    v_mov_b32_e32 v3, 0
; HSA-NEXT:    v_mov_b32_e32 v5, 0
; HSA-NEXT:    v_mov_b32_e32 v7, 0
; HSA-NEXT:    v_mov_b32_e32 v9, 0
; HSA-NEXT:    v_mov_b32_e32 v11, 0
; HSA-NEXT:    v_mov_b32_e32 v13, 0
; HSA-NEXT:    v_mov_b32_e32 v15, 0
; HSA-NEXT:    v_mov_b32_e32 v17, 0
; HSA-NEXT:    v_mov_b32_e32 v19, 0
; HSA-NEXT:    v_mov_b32_e32 v21, 0
; HSA-NEXT:    v_mov_b32_e32 v23, 0
; HSA-NEXT:    v_mov_b32_e32 v25, 0
; HSA-NEXT:    v_mov_b32_e32 v27, 0
; HSA-NEXT:    v_mov_b32_e32 v29, 0
; HSA-NEXT:    v_mov_b32_e32 v31, 0
; HSA-NEXT:    s_setpc_b64 s[30:31]
  %cast = addrspacecast <16 x ptr addrspace(6)> %ptr to <16 x ptr>
  ret <16 x ptr> %cast
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind "amdgpu-32bit-address-high-bits"="0xffff8000" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
