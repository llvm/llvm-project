; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCNO,COV5O %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCNO,COV5O %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=GCNO,COV4O %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -attributor-assume-closed-world=false -mcpu=gfx900 | FileCheck -check-prefixes=GCNC,COV5C %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -attributor-assume-closed-world=false -mcpu=gfx900 | FileCheck -check-prefixes=GCNC,COV5C %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -attributor-assume-closed-world=false -mcpu=gfx900 | FileCheck -check-prefixes=GCNC,COV4C %s

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

; No stack objects, only indirect call has to enable scratch
; GCNO-LABEL: test_indirect_call:
; GCNC-LABEL: test_indirect_call:

; COV5O: .amdhsa_private_segment_fixed_size 0{{$}}
; COV5C: .amdhsa_private_segment_fixed_size 0{{$}}
; COV4C: .amdhsa_private_segment_fixed_size 0{{$}}
; COV4O: .amdhsa_private_segment_fixed_size 16384{{$}}

; GCNO: .amdhsa_user_sgpr_private_segment_buffer 1
; GCNC: .amdhsa_user_sgpr_private_segment_buffer 1

; COV5O: .amdhsa_uses_dynamic_stack 1
; COV5C: .amdhsa_uses_dynamic_stack 0
; GCNO: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; GCNC: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
define amdgpu_kernel void @test_indirect_call() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
