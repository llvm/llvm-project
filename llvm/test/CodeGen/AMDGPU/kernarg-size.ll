; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck --check-prefix=DOORBELL %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck --check-prefix=DOORBELL %s

declare void @llvm.trap() #0

; DOORBELL:      .amdhsa_kernel trap
; DOORBELL-NEXT:     .amdhsa_group_segment_fixed_size 0
; DOORBELL-NEXT:     .amdhsa_private_segment_fixed_size 0
; DOORBELL-NEXT:     .amdhsa_kernarg_size 8
; DOORBELL-NEXT:     .amdhsa_user_sgpr_count 6
; DOORBELL-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; DOORBELL:      .end_amdhsa_kernel

define amdgpu_kernel void @trap(ptr addrspace(1) nocapture readonly %arg0) {
  store volatile i32 1, ptr addrspace(1) %arg0
  call void @llvm.trap()
  unreachable
  store volatile i32 2, ptr addrspace(1) %arg0
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
