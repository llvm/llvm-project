; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}lower_control_flow_unreachable_terminator:
; GCN: v_cmp_eq_u32
; GCN: s_and_saveexec_b64
; GCN: s_xor_b64
; GCN: ; mask branch [[RET:BB[0-9]+]]
; GCN: s_branch [[UNREACHABLE:BB[0-9]+_[0-9]+]]

; GCN: [[RET]]
; GCN: s_or_b64 exec, exec
; GCN: s_endpgm

; GCN: [[UNREACHABLE]]:
; GCN: ds_write_b32
; GCN: s_waitcnt
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
; GCN: v_cmp_eq_u32
; GCN: s_and_saveexec_b64
; GCN: s_xor_b64
; GCN: ; mask branch [[UNREACHABLE:BB[0-9]+_[0-9]+]]

; GCN-NEXT: ; %ret
; GCN-NEXT: s_endpgm

; GCN-NEXT: [[UNREACHABLE]]:
; GCN-NEXT: s_or_b64 exec, exec
; GCN: ds_write_b32
; GCN: s_waitcnt
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

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
