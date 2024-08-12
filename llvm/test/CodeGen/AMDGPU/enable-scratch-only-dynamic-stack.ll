; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCN,COV5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCN,COV5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCN,COV4 %s

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

; No stack objects, only indirect call has to enable scrathch
; GCN-LABEL: test_indirect_call:

; GCN: .amdhsa_private_segment_fixed_size test_indirect_call.private_seg_size
; GCN: .amdhsa_user_sgpr_private_segment_buffer 1
; COV5: .amdhsa_uses_dynamic_stack ((59|((test_indirect_call.has_dyn_sized_stack|test_indirect_call.has_recursion)<<11))&2048)>>11
; COV5: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(test_indirect_call.private_seg_size*64, 1024))/1024)>0)||(test_indirect_call.has_dyn_sized_stack|test_indirect_call.has_recursion))|5016)&1
; COV4: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(test_indirect_call.private_seg_size*64, 1024))/1024)>0)||(test_indirect_call.has_dyn_sized_stack|test_indirect_call.has_recursion))|5020)&1

; COV5: .set test_indirect_call.private_seg_size, 0{{$}}
; COV4: .set test_indirect_call.private_seg_size, 0+(max(16384))
; COV5: .set test_indirect_call.has_recursion, 1
; COV5: .set test_indirect_call.has_indirect_call, 1

define amdgpu_kernel void @test_indirect_call() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
