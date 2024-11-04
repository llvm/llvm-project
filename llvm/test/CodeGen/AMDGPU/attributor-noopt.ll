; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefix=OPT %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=NOOPT,COV4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=NOOPT,COV5 %s

; Check that AMDGPUAttributor is not run with -O0.
; OPT: .amdhsa_user_sgpr_private_segment_buffer 1
; OPT: .amdhsa_user_sgpr_dispatch_ptr 0
; OPT: .amdhsa_user_sgpr_queue_ptr 0
; OPT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; OPT: .amdhsa_user_sgpr_dispatch_id 0
; OPT: .amdhsa_user_sgpr_flat_scratch_init 0
; OPT: .amdhsa_user_sgpr_private_segment_size 0
; OPT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; OPT: .amdhsa_system_sgpr_workgroup_id_x 1
; OPT: .amdhsa_system_sgpr_workgroup_id_y 0
; OPT: .amdhsa_system_sgpr_workgroup_id_z 0
; OPT: .amdhsa_system_sgpr_workgroup_info 0
; OPT: .amdhsa_system_vgpr_workitem_id 0

; NOOPT: .amdhsa_user_sgpr_private_segment_buffer 1
; NOOPT: .amdhsa_user_sgpr_dispatch_ptr 1
; COV4: .amdhsa_user_sgpr_queue_ptr 1
; COV5: .amdhsa_user_sgpr_queue_ptr 0
; NOOPT: .amdhsa_user_sgpr_kernarg_segment_ptr 1
; NOOPT: .amdhsa_user_sgpr_dispatch_id 1
; NOOPT: .amdhsa_user_sgpr_flat_scratch_init 0
; NOOPT: .amdhsa_user_sgpr_private_segment_size 0
; NOOPT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; NOOPT: .amdhsa_system_sgpr_workgroup_id_x 1
; NOOPT: .amdhsa_system_sgpr_workgroup_id_y 1
; NOOPT: .amdhsa_system_sgpr_workgroup_id_z 1
; NOOPT: .amdhsa_system_sgpr_workgroup_info 0
; NOOPT: .amdhsa_system_vgpr_workitem_id 2
define amdgpu_kernel void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 CODE_OBJECT_VERSION}
