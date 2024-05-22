; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - | FileCheck -check-prefix=GCN %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - | FileCheck -check-prefix=GCN-V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - | FileCheck -check-prefix=GCN-V5 %s

; Make sure there's no assertion when trying to report the resource
; usage for a function which becomes dead during codegen.

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

; GCN-LABEL: unreachable:
; Function info:
; codeLenInByte = 4
define internal fastcc void @unreachable() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  unreachable
}


; GCN-LABEL: entry:
; GCN-NOT: s_swappc_b64
; GCN: s_endpgm

; GCN: .amdhsa_private_segment_fixed_size 0
; GCN-NOT: .amdhsa_uses_dynamic_stack 0
; GCN-V5: .amdhsa_uses_dynamic_stack 0
define amdgpu_kernel void @entry() {
bb0:
  br i1 false, label %bb1, label %bb2

bb1:
  tail call fastcc void @unreachable()
  unreachable

bb2:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
