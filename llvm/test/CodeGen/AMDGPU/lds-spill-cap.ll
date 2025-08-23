; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-lds-spill-bytes=256 < %s \
; RUN:   | FileCheck --check-prefix=CAP %s

; Test 1: Auto-cap LDS spill budget when kernel's own LDS is large.
;
; This kernel uses 60 KB (15360 x i32 = 61440 bytes) of static LDS.
; With max_workgroup_size=1024 and hardware limit of 64 KB (65536 bytes),
; available = 65536 - 61440 = 4096 bytes.
; Max per-thread = 4096 / 1024 = 4 bytes (1 dword slot). The requested
; 256 bytes/thread gets auto-capped to 4.
; Total LDS = 61440 + 4 * 1024 = 65536 (exactly at the hardware limit).
; Overflow spills use scratch.

; CAP: lds_spill_auto_cap:
; LDS spill is used (capped to 1 slot):
; CAP: ds_store_b32
; Overflow spills go to scratch:
; CAP: scratch_store_b{{32|128}}
; Total LDS stays within the 64 KB hardware limit:
; CAP: .amdhsa_group_segment_fixed_size 65536
; CAP: .amdhsa_private_segment_fixed_size {{[1-9][0-9]*}}

@lds_large = internal unnamed_addr addrspace(3) global [15360 x i32] poison, align 4

define amdgpu_kernel void @lds_spill_auto_cap(ptr addrspace(1) %p) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <8 x float>, ptr addrspace(1) %p, i32 %tid
  %v = load volatile <8 x float>, ptr addrspace(1) %gep
  %lds.gep = getelementptr inbounds [15360 x i32], ptr addrspace(3) @lds_large, i32 0, i32 %tid
  store i32 42, ptr addrspace(3) %lds.gep, align 4
  store volatile <8 x float> %v, ptr addrspace(1) poison
  ret void
}

; Test 2: Dynamic LDS disables LDS spilling. The kernel uses an external
; zero-sized LDS global (dynamic LDS marker). The compiler disables LDS
; spilling because dynamic LDS size is unknown at compile time.
; All spills go to scratch; group segment is zero (no static LDS allocation).

; CAP: lds_spill_dynamic_lds:
; All spills use scratch:
; CAP: scratch_store_b32
; No LDS spill reservation:
; CAP: .amdhsa_group_segment_fixed_size 0
; CAP: .amdhsa_private_segment_fixed_size {{[1-9][0-9]*}}

@lds_dynamic = external unnamed_addr addrspace(3) global [0 x i32], align 4

define amdgpu_kernel void @lds_spill_dynamic_lds(ptr addrspace(1) %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <8 x float>, ptr addrspace(1) %p, i32 %tid
  %v = load volatile <8 x float>, ptr addrspace(1) %gep
  %dyn.gep = getelementptr inbounds [0 x i32], ptr addrspace(3) @lds_dynamic, i32 0, i32 %tid
  store i32 99, ptr addrspace(3) %dyn.gep, align 4
  store volatile <8 x float> %v, ptr addrspace(1) poison
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { nounwind "amdgpu-num-vgpr"="6" "amdgpu-flat-work-group-size"="1024,1024" }
attributes #1 = { nounwind "amdgpu-num-vgpr"="6" "amdgpu-flat-work-group-size"="64,64" }
