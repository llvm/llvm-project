; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=HSA %s

; HSA-LABEL: {{^}}use_group_to_flat_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; HSA: enable_sgpr_queue_ptr = 1

; HSA-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; HSA-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10{{$}}

; HSA-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[APERTURE]]
; HSA-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]

; HSA-DAG: v_cmp_ne_u32_e64 vcc, [[PTR]], -1
; HSA-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]]
; HSA-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7

; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, [[K]]
define void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %stof
  ret void
}

; HSA-LABEL: {{^}}use_private_to_flat_addrspacecast:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; HSA: enable_sgpr_queue_ptr = 1

; HSA-DAG: s_load_dword [[PTR:s[0-9]+]], s[6:7], 0x0{{$}}
; HSA-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x11{{$}}

; HSA-DAG: v_mov_b32_e32 [[VAPERTURE:v[0-9]+]], [[APERTURE]]
; HSA-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]

; HSA-DAG: v_cmp_ne_u32_e64 vcc, [[PTR]], -1
; HSA-DAG: v_cndmask_b32_e32 v[[HI:[0-9]+]], 0, [[VAPERTURE]]
; HSA-DAG: v_cndmask_b32_e32 v[[LO:[0-9]+]], 0, [[VPTR]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7

; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, [[K]]
define void @use_private_to_flat_addrspacecast(i32* %ptr) #0 {
  %stof = addrspacecast i32* %ptr to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %stof
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
define void @use_global_to_flat_addrspacecast(i32 addrspace(1)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(1)* %ptr to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %stof
  ret void
}

; no-op
; HSA-LABEl: {{^}}use_constant_to_flat_addrspacecast:
; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA: flat_load_dword v{{[0-9]+}}, v{{\[}}[[VPTRLO]]:[[VPTRHI]]{{\]}}
define void @use_constant_to_flat_addrspacecast(i32 addrspace(2)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(2)* %ptr to i32 addrspace(4)*
  %ld = load volatile i32, i32 addrspace(4)* %stof
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
define void @use_flat_to_group_addrspacecast(i32 addrspace(4)* %ptr) #0 {
  %ftos = addrspacecast i32 addrspace(4)* %ptr to i32 addrspace(3)*
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
; HSA-DAG: v_cndmask_b32_e32 [[CASTPTR:v[0-9]+]], -1, v[[VPTR_LO]]
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 0{{$}}
; HSA: buffer_store_dword v[[K]], [[CASTPTR]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}
define void @use_flat_to_private_addrspacecast(i32 addrspace(4)* %ptr) #0 {
  %ftos = addrspacecast i32 addrspace(4)* %ptr to i32*
  store volatile i32 0, i32* %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_global_addrspacecast:
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}, s[4:5], 0x0
; HSA-DAG: v_mov_b32_e32 v[[VPTRLO:[0-9]+]], s[[PTRLO]]
; HSA-DAG: v_mov_b32_e32 v[[VPTRHI:[0-9]+]], s[[PTRHI]]
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0
; HSA: flat_store_dword v{{\[}}[[VPTRLO]]:[[VPTRHI]]{{\]}}, [[K]]
define void @use_flat_to_global_addrspacecast(i32 addrspace(4)* %ptr) #0 {
  %ftos = addrspacecast i32 addrspace(4)* %ptr to i32 addrspace(1)*
  store volatile i32 0, i32 addrspace(1)* %ftos
  ret void
}

; HSA-LABEL: {{^}}use_flat_to_constant_addrspacecast:
; HSA: enable_sgpr_queue_ptr = 0

; HSA: s_load_dwordx2 s{{\[}}[[PTRLO:[0-9]+]]:[[PTRHI:[0-9]+]]{{\]}}, s[4:5], 0x0
; HSA: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTRLO]]:[[PTRHI]]{{\]}}, 0x0
define void @use_flat_to_constant_addrspacecast(i32 addrspace(4)* %ptr) #0 {
  %ftos = addrspacecast i32 addrspace(4)* %ptr to i32 addrspace(2)*
  load volatile i32, i32 addrspace(2)* %ftos
  ret void
}

; HSA-LABEL: {{^}}cast_0_group_to_flat_addrspacecast:
; HSA: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE]]
; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define void @cast_0_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(3)* null to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define void @cast_0_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(4)* null to i32 addrspace(3)*
  store volatile i32 7, i32 addrspace(3)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_group_to_flat_addrspacecast:
; HSA: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define void @cast_neg1_group_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(3)* inttoptr (i32 -1 to i32 addrspace(3)*) to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_neg1_flat_to_group_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: ds_write_b32 [[PTR]], [[K]]
define void @cast_neg1_flat_to_group_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(4)* inttoptr (i64 -1 to i32 addrspace(4)*) to i32 addrspace(3)*
  store volatile i32 7, i32 addrspace(3)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_private_to_flat_addrspacecast:
; HSA: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x11
; HSA-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE]]
; HSA-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; HSA-DAG: v_mov_b32_e32 v[[K:[0-9]+]], 7{{$}}
; HSA: flat_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}, v[[K]]
define void @cast_0_private_to_flat_addrspacecast() #0 {
  %cast = addrspacecast i32* null to i32 addrspace(4)*
  store volatile i32 7, i32 addrspace(4)* %cast
  ret void
}

; HSA-LABEL: {{^}}cast_0_flat_to_private_addrspacecast:
; HSA-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], -1{{$}}
; HSA-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 7{{$}}
; HSA: buffer_store_dword [[K]], [[PTR]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
define void @cast_0_flat_to_private_addrspacecast() #0 {
  %cast = addrspacecast i32 addrspace(4)* null to i32 addrspace(0)*
  store volatile i32 7, i32* %cast
  ret void
}

; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.

; HSA-LABEL: {{^}}branch_use_flat_i32:
; HSA: flat_store_dword {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}
; HSA: s_endpgm
define void @branch_use_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* %gptr, i32 addrspace(3)* %lptr, i32 %x, i32 %c) #0 {
entry:
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %local, label %global

local:
  %flat_local = addrspacecast i32 addrspace(3)* %lptr to i32 addrspace(4)*
  br label %end

global:
  %flat_global = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  br label %end

end:
  %fptr = phi i32 addrspace(4)* [ %flat_local, %local ], [ %flat_global, %global ]
  store volatile i32 %x, i32 addrspace(4)* %fptr, align 4
;  %val = load i32, i32 addrspace(4)* %fptr, align 4
;  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; Check for prologue initializing special SGPRs pointing to scratch.
; HSA-LABEL: {{^}}store_flat_scratch:
; HSA-DAG: s_mov_b32 flat_scratch_lo, s9
; HSA-DAG: s_add_u32 [[ADD:s[0-9]+]], s8, s11
; HSA: s_lshr_b32 flat_scratch_hi, [[ADD]], 8
; HSA: flat_store_dword
; HSA: s_barrier
; HSA: flat_load_dword
define void @store_flat_scratch(i32 addrspace(1)* noalias %out, i32) #0 {
  %alloca = alloca i32, i32 9, align 4
  %x = call i32 @llvm.amdgcn.workitem.id.x() #2
  %pptr = getelementptr i32, i32* %alloca, i32 %x
  %fptr = addrspacecast i32* %pptr to i32 addrspace(4)*
  store volatile i32 %x, i32 addrspace(4)* %fptr
  ; Dummy call
  call void @llvm.amdgcn.s.barrier() #1
  %reload = load volatile i32, i32 addrspace(4)* %fptr, align 4
  store volatile i32 %reload, i32 addrspace(1)* %out, align 4
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind readnone }
