; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O2 | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefix=OPT %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=NOOPT,COV4 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=NOOPT,COV5 %s

; Check that AMDGPUAttributor is not run with -O0.
; OPT: .amdhsa_user_sgpr_private_segment_buffer 1
; OPT: .amdhsa_user_sgpr_dispatch_ptr 0
; OPT: .amdhsa_user_sgpr_queue_ptr 0
; OPT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; OPT: .amdhsa_user_sgpr_dispatch_id 0
; OPT: .amdhsa_user_sgpr_flat_scratch_init 0
; OPT: .amdhsa_user_sgpr_private_segment_size 0
; OPT: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(foo.private_seg_size*64, 1024))/1024)>0)||(foo.has_dyn_sized_stack|foo.has_recursion))|136)&1
; OPT: .amdhsa_system_sgpr_workgroup_id_x 1
; OPT: .amdhsa_system_sgpr_workgroup_id_y 0
; OPT: .amdhsa_system_sgpr_workgroup_id_z 0
; OPT: .amdhsa_system_sgpr_workgroup_info 0
; OPT: .amdhsa_system_vgpr_workitem_id 0
; OPT: .set foo.num_vgpr, 0
; OPT: .set foo.num_agpr, 0
; OPT: .set foo.num_sgpr, 0
; OPT: .set foo.private_seg_size, 0
; OPT: .set foo.uses_vcc, 0
; OPT: .set foo.uses_flat_scratch, 0
; OPT: .set foo.has_dyn_sized_stack, 0
; OPT: .set foo.has_recursion, 0
; OPT: .set foo.has_indirect_call, 0

; NOOPT: .amdhsa_user_sgpr_private_segment_buffer 1
; NOOPT: .amdhsa_user_sgpr_dispatch_ptr 1
; COV4: .amdhsa_user_sgpr_queue_ptr 1
; COV5: .amdhsa_user_sgpr_queue_ptr 0
; NOOPT: .amdhsa_user_sgpr_kernarg_segment_ptr 1
; NOOPT: .amdhsa_user_sgpr_dispatch_id 1
; NOOPT: .amdhsa_user_sgpr_flat_scratch_init 0
; NOOPT: .amdhsa_user_sgpr_private_segment_size 0
; COV4: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(foo.private_seg_size*64, 1024))/1024)>0)||(foo.has_dyn_sized_stack|foo.has_recursion))|5016)&1
; COV5: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(foo.private_seg_size*64, 1024))/1024)>0)||(foo.has_dyn_sized_stack|foo.has_recursion))|5012)&1
; NOOPT: .amdhsa_system_sgpr_workgroup_id_x 1
; NOOPT: .amdhsa_system_sgpr_workgroup_id_y 1
; NOOPT: .amdhsa_system_sgpr_workgroup_id_z 1
; COV4: .amdhsa_system_sgpr_workgroup_info 0
; COV5: .amdhsa_system_sgpr_workgroup_info 0
; COV4: .amdhsa_system_vgpr_workitem_id 2
; COV5: .amdhsa_system_vgpr_workitem_id 2
; NOOPT: .set foo.num_vgpr, 0
; NOOPT: .set foo.num_agpr, 0
; NOOPT: .set foo.num_sgpr, 0
; NOOPT: .set foo.private_seg_size, 0
; NOOPT: .set foo.uses_vcc, 0
; NOOPT: .set foo.uses_flat_scratch, 0
; NOOPT: .set foo.has_dyn_sized_stack, 0
; NOOPT: .set foo.has_recursion, 0
; NOOPT: .set foo.has_indirect_call, 0

define amdgpu_kernel void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
