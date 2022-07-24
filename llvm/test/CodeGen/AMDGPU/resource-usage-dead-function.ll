; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - %s | FileCheck -check-prefix=GCN %s

; Make sure there's no assertion when trying to report the resource
; usage for a function which becomes dead during codegen.

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant void()*, align 4

; GCN-LABEL: unreachable:
; Function info:
; codeLenInByte = 4
define internal fastcc void @unreachable() {
  %fptr = load void()*, void()* addrspace(4)* @gv.fptr0
  call void %fptr()
  unreachable
}


; GCN-LABEL: entry:
; GCN-NOT: s_swappc_b64
; GCN: s_endpgm

; GCN: .amdhsa_private_segment_fixed_size 0
; GCN: .amdhsa_uses_dynamic_stack 0
define amdgpu_kernel void @entry() {
bb0:
  br i1 false, label %bb1, label %bb2

bb1:
  tail call fastcc void @unreachable()
  unreachable

bb2:
  ret void
}
