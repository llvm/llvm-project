; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 < %s | FileCheck -check-prefixes=GCN,COV5 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 < %s | FileCheck -check-prefixes=GCN,COV4 %s

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

; No stack objects, only indirect call has to enable scrathch
; GCN-LABEL: test_indirect_call:

; COV5: .amdhsa_private_segment_fixed_size 0{{$}}
; COV4: .amdhsa_private_segment_fixed_size 16384{{$}}

; GCN: .amdhsa_user_sgpr_private_segment_buffer 1

; COV5: .amdhsa_uses_dynamic_stack 1
; GCN: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
define amdgpu_kernel void @test_indirect_call() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}
