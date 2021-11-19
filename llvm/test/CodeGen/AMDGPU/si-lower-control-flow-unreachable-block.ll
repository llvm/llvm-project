; RUN: llc -march=amdgcn -verify-machineinstrs -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}lower_control_flow_unreachable_terminator:
; GCN: v_cmp_eq_u32
; GCN: s_and_saveexec_b64
; GCN-NEXT: s_cbranch_execz .LBB0_{{[0-9]+}}

; GCN-NEXT: ; %bb.{{[0-9]+}}: ; %unreachable
; GCN: ds_write_b32
; GCN: ; divergent unreachable

; GCN-NEXT: BB0_{{[0-9]+}}: ; %UnifiedReturnBlock
; GCN: s_endpgm

define amdgpu_kernel void @lower_control_flow_unreachable_terminator() #0 {
bb:
  %tmp15 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp63 = icmp eq i32 %tmp15, 32
  br i1 %tmp63, label %unreachable, label %ret

unreachable:
  store volatile i32 0, i32 addrspace(3)* undef, align 4
  unreachable

ret:
  ret void
}

; GCN-LABEL: {{^}}lower_control_flow_unreachable_terminator_swap_block_order:
; GCN: v_cmp_ne_u32
; GCN: s_and_saveexec_b64
; GCN-NEXT: s_cbranch_execz .LBB1_{{[0-9]+}}

; GCN-NEXT: ; %bb.{{[0-9]+}}: ; %unreachable
; GCN: ds_write_b32
; GCN: ; divergent unreachable

; GCN: BB1_{{[0-9]+}}:
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @lower_control_flow_unreachable_terminator_swap_block_order() #0 {
bb:
  %tmp15 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp63 = icmp eq i32 %tmp15, 32
  br i1 %tmp63, label %ret, label %unreachable

ret:
  ret void

unreachable:
  store volatile i32 0, i32 addrspace(3)* undef, align 4
  unreachable
}

; GCN-LABEL: {{^}}uniform_lower_control_flow_unreachable_terminator:
; GCN: s_cmp_lg_u32
; GCN: s_cbranch_scc0 [[UNREACHABLE:.LBB[0-9]+_[0-9]+]]

; GCN-NEXT: %bb.{{[0-9]+}}: ; %ret
; GCN-NEXT: s_endpgm

; GCN: [[UNREACHABLE]]:
; GCN: ds_write_b32
define amdgpu_kernel void @uniform_lower_control_flow_unreachable_terminator(i32 %arg0) #0 {
bb:
  %tmp63 = icmp eq i32 %arg0, 32
  br i1 %tmp63, label %unreachable, label %ret

unreachable:
  store volatile i32 0, i32 addrspace(3)* undef, align 4
  unreachable

ret:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
